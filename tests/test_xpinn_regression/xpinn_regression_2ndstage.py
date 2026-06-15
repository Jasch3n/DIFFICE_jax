import argparse
import os
import pickle
import re
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from kfac_jax import loss_functions as kfac_loss_functions

import xpinn_regression as base
from diffice_jax import loss_regression_2ndstage_xpinn


SECOND_STAGE_EMBEDDING_CONFIG = [
        dict(embedding_u=True, embedding_mu=False, embedding_c=False, embed_n=64, embed_std=1.0),
        dict(embedding_u=True, embedding_mu=False, embedding_c=False, embed_n=32, embed_std=2.0),
]
SECOND_STAGE_EMBEDDING_CONFIG = None
SECOND_STAGE_NETWORK_CONFIG = [
    base.net_arch(u=(3, 20), mu=(1, 10), c0=(1, 10)),
    base.net_arch(u=(1, 10), mu=(1, 10), c0=(1, 10)),
]
CKPT_PATH = f'{base.CKPT_PATH}_stage2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-stage-checkpoint', required=True)
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    return parser.parse_args()


def load_first_stage_params(first_stage_checkpoint: str):
    ckpt_path = os.path.abspath(first_stage_checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if os.path.commonpath([base.REGRESSION_ROOT, ckpt_path]) != base.REGRESSION_ROOT:
        raise ValueError(f"Checkpoint must be inside {base.REGRESSION_ROOT}: {ckpt_path}")

    ckpt_name = os.path.basename(ckpt_path)
    if re.fullmatch(r'(step|KFAC_step)_\d+\.pkl', ckpt_name) is None:
        raise ValueError(f"First-stage checkpoint must match step_[num].pkl or KFAC_step_[num].pkl: {ckpt_name}")

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    if 'params' not in ckpt:
        raise ValueError(f"Checkpoint missing required key: params")

    step = int(ckpt.get('step', -1))
    print(f"[stage2] Loaded first-stage checkpoint: {ckpt_path} (step {step})")
    return jax.device_put(ckpt['params'])


def build_second_stage_data_output(data_output: base.DataOutput, sol_NN, first_stage_params):
    data_all, _, idxgall, _ = data_output
    predNN, _ = sol_NN
    second_stage_data_all = list(data_all)
    residual_rms = {}
    for idx in idxgall:
        x_all = data_all[idx][0][0]
        u_all = data_all[idx][1][0]
        u_first = jax.lax.stop_gradient(predNN(first_stage_params, x_all, idx)[:, 0:2])
        u_res = u_all - u_first
        u_rms = jnp.maximum(jnp.sqrt(jnp.mean(u_res ** 2, axis=0)), 1e-12)
        u_target = u_res / u_rms
        region_data = list(second_stage_data_all[idx])
        region_u = list(region_data[1])
        region_u[0] = u_target
        region_data[1] = tuple(region_u) if isinstance(region_data[1], tuple) else region_u
        second_stage_data_all[idx] = tuple(region_data) if isinstance(second_stage_data_all[idx], tuple) else region_data
        residual_rms[idx] = u_rms
        mean_abs_u = float(jnp.mean(jnp.abs(u_res[:, 0])))
        mean_abs_v = float(jnp.mean(jnp.abs(u_res[:, 1])))
        rms_u = float(u_rms[0])
        rms_v = float(u_rms[1])
        print(f'[stage2] Region {idx}: precomputed {x_all.shape[0]} velocity residual samples '
              f'| mean_abs_u={mean_abs_u:.3e} | mean_abs_v={mean_abs_v:.3e} '
              f'| rms_u={rms_u:.3e} | rms_v={rms_v:.3e}')
    return base.DataOutput(second_stage_data_all, data_output.basal_mask, data_output.idxgall, data_output.scale), residual_rms


def make_floating_region_freeze_spec(params, basal_mask):
    return {
        family: [
            idx for idx, subnet in enumerate(subnets)
            if subnet is not None and not basal_mask[idx]
        ]
        for family, subnets in params.items()
    }


def initialize_loss(keys, data_output: base.DataOutput, xpinn_output: base.XPINNOutput) -> base.LossOutput:
    data_all, basal_mask, idxgall, scale = data_output
    params, sol_NN, eval_f = xpinn_output

    n_pt = base.REGRESSION_N_PT_BY_NSUB[len(idxgall)]
    data_f = base.dsample_regression_xpinn(data_all, idxgall, n_pt, basal_mask=basal_mask)

    dkey, keys = random.split(keys, 2)
    smp = data_f(dkey)

    NN_loss = loss_regression_2ndstage_xpinn(sol_NN, idxgall, basal_mask=basal_mask)

    initial_loss = NN_loss(params, smp)[0]
    NN_loss.lref = initial_loss
    return keys, base.LossOutput(data_f, NN_loss, initial_loss, sol_NN)


def kfac_optimize(keys, kfac_config, data: base.DataOutput, xpinn: base.XPINNOutput,
                  loss: base.LossOutput, resume_state: base.ResumeState | None = None):
    lossf = loss.loss_f
    dataf = loss.data_f
    initial_params = xpinn.params if resume_state is None else resume_state.params
    freeze_mask, freeze_spec = base.make_freeze_mask(
        initial_params, make_floating_region_freeze_spec(initial_params, data.basal_mask))
    frozen_params = initial_params
    print(f'[stage2] KFAC freeze spec: {freeze_spec}')

    def kfac_lossf(params, data_batch):
        active_params = base.apply_frozen_params(params, frozen_params, freeze_mask)
        loss_n, loss_info, _ = lossf(active_params, data_batch)
        residuals = lossf.kfac_residuals(active_params, data_batch) / jnp.sqrt(lossf.lref)
        kfac_loss_functions.register_squared_error_loss(
            residuals,
            targets=jnp.zeros_like(residuals),
        )
        return loss_n, loss_info

    optim = base.KfacOptimizer(loss_fn=kfac_lossf, **kfac_config).get_optimizer()

    rng, init_key = random.split(keys)
    init_data = dataf(init_key)
    reset_opt_state = resume_state is not None and base.freeze_spec_changed(resume_state.freeze_spec, freeze_spec)
    if resume_state is None:
        params = xpinn.params
        opt_state = optim.init(params, init_key, init_data)
        kfac_start_step = 0
    else:
        params = resume_state.params
        opt_state = optim.init(params, init_key, init_data) if reset_opt_state else resume_state.opt_state
        kfac_start_step = resume_state.step + 1
        if reset_opt_state:
            print('[resume] Freeze spec changed; reinitialized KFAC optimizer state.')
    params = base.project_frozen_params(params, frozen_params, freeze_mask)
    loss_all = [] if resume_state is None or resume_state.loss_history is None else list(resume_state.loss_history)
    damping = kfac_config['initial_damping'] if resume_state is None or resume_state.damping is None else resume_state.damping
    damping_decay = kfac_config['damping_adaptation_decay']
    damping_min = kfac_config['min_damping']

    def save_checkpoint(step, params, opt_state, damping):
        ckpt = {
            'step': step,
            'params': jax.device_get(params),
            'opt_state': jax.device_get(opt_state),
            'damping': jax.device_get(damping),
            'loss_history': jax.device_get(loss_all),
            'freeze_spec': freeze_spec,
        }
        with open(os.path.join(CKPT_PATH, f'KFAC_step_{step}.pkl'), 'wb') as f:
            pickle.dump(ckpt, f)

    def print_subregion_errs(reg_err_list):
        data_err = reg_err_list[0]
        n_region = len(data.idxgall)
        for idx in range(len(data_err)):
            subregion_data_errs = data_err[idx]
            print(f'                  Region {idx} ({'grounded' if data.basal_mask[idx] else 'floating'}): '
                  f'u_total_misfit_normalized_mse = {subregion_data_errs[0]:<5.3e} | '
                  f'v_total_misfit_normalized_mse = {subregion_data_errs[1]:<5.3e} | '
                  f'h = {subregion_data_errs[2]:<5.3e} | s = {subregion_data_errs[3]:<5.3e} | '
                  f'mu= {subregion_data_errs[4]:<5.3e} | c = {subregion_data_errs[5]:<5.3e}')

    for step in range(kfac_start_step, base.KFAC_MAXITER):
        rng, step_key, data_key = random.split(rng, 3)
        data_batch = dataf(data_key)
        params, opt_state, stats = optim.step(
            params, opt_state, step_key, batch=data_batch, damping=damping, global_step_int=step)
        params = base.project_frozen_params(params, frozen_params, freeze_mask)

        loss_info = stats['aux']
        loss_all.append(loss_info)

        if step % base.KFAC_LOG_RATE == 0:
            _, loss_info, reg_err_list = loss.loss_f(
                base.apply_frozen_params(params, frozen_params, freeze_mask), data_batch)
            dmp = stats['damping']
            print(f'--------------------------------- STEP {step} ------------------------------------')
            print(f'STAGE2 KFAC step {step}: loss={loss_info[0]:.2e} | '
                  f'u_total_misfit_normalized_mse={loss_info[1]:.2e} | '
                  f'v_total_misfit_normalized_mse={loss_info[2]:.2e} | '
                  f'h={loss_info[3]:.2e} | s={loss_info[4]:.2e} | '
                  f'mu={loss_info[5]:.2e} | C={loss_info[6]:.2e} | Dp={dmp:.2e}')
            print_subregion_errs(reg_err_list)

        if step % base.KFAC_CKPT_RATE == 0:
            save_checkpoint(step, params, opt_state, damping)

        if damping > damping_min:
            damping *= damping_decay

    return params, loss_all


if __name__ == '__main__':
    wall_start = time.time()
    args = parse_args()
    os.makedirs(CKPT_PATH, exist_ok=True)
    keys = random.PRNGKey(8132002)

    print('[stage2] First-stage checkpoint supplied; training a brand new second-stage XPINN.')
    print('[stage2] Second-stage velocity data term fits observed velocity minus first-stage velocity.')
    print('[stage2] Stage-two network predicts velocity residual divided by stage-one residual RMS.')
    print('[stage2] Stage-two logs report mean-squared normalized total velocity misfit.')
    print(f'[stage2] Second-stage embedding config: {SECOND_STAGE_EMBEDDING_CONFIG}')
    print(f'[stage2] Second-stage network config: {SECOND_STAGE_NETWORK_CONFIG}')
    print(f'[stage2] Second-stage checkpoints will be saved to: {CKPT_PATH}')

    data_output = base.load_data(base.DATA_PATH)
    keys, xpinn_output = base.initialize_xpinn(
        keys, data_output,
        embedding_config=SECOND_STAGE_EMBEDDING_CONFIG,
        network_config=SECOND_STAGE_NETWORK_CONFIG)
    print('[stage2] Initialized fresh second-stage network parameters.')

    first_stage_params = load_first_stage_params(args.first_stage_checkpoint)
    data_output, residual_rms = build_second_stage_data_output(
        data_output, xpinn_output.sol_NN, first_stage_params)

    resume_state = base.load_checkpoint(args.resume_checkpoint)
    if resume_state is not None:
        print('[stage2] Resuming second-stage trainable parameters from resume checkpoint.')
        xpinn_output = base.XPINNOutput(resume_state.params, xpinn_output.sol_NN, xpinn_output.eval_f)

    keys, loss_output = initialize_loss(keys, data_output, xpinn_output)
    print(f'Initial second-stage X-PINN normalized total velocity misfit loss = {loss_output.initial_loss:.2e}')

    params, loss_history = kfac_optimize(
        keys, base.kfac_config, data_output, xpinn_output, loss_output,
        resume_state=resume_state)
    with open(os.path.join(CKPT_PATH, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(loss_history, f)
    wall_elapsed = time.time() - wall_start
    print(f'Total wall time: {wall_elapsed:.2f} s ({wall_elapsed / 60:.2f} min)')
