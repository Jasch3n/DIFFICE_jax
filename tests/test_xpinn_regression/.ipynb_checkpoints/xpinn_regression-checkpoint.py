import jax.numpy as jnp
import pickle
import jax
from jax import config

#[NOTE]: Double-precision floating point arithmetic can be importantwhen equation loss is small.
config.update("jax_enable_x64", True)
if jax.default_backend() == 'METAL':
    config.update("jax_enable_x64", False)

import random as pyrnd
from jax import random
import orbax.checkpoint as ocp
from flax.training import train_state 
from jax.tree_util import tree_map
from scipy.io import loadmat
import time, os, pickle, argparse
import re
import optax

from diffice_jax import normdata_xpinn, dsample_regression_xpinn
import diffice_jax
print(f"[1] DIFFICE_jax path at {diffice_jax.__file__}")

from diffice_jax import vectgrad, ssa_iso, dbc_iso
from diffice_jax import init_xpinn, solu_xpinn
from diffice_jax import loss_iso_xpinn
from diffice_jax import adam_opt, lbfgs_opt
from diffice_jax import loss_regression_xpinn

from diffice_jax.data.xpinns.preprocessing import SubScaleResult

from datetime import datetime
import json, copy
from typing import NamedTuple, Any, List, Tuple, Callable
from jax.typing import ArrayLike

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REGRESSION_ROOT = os.path.join(PROJECT_ROOT, 'test_xpinn_regression')
LBFGS_TOL = 1e-6
LBFGS_LOG_RATE = 100

#######################################################################
TEST_CASE = 'NO_RUMPLE'
assert TEST_CASE in ['RUMPLE', 'NO_RUMPLE'], "TEST_CASE must be one of ['RUMPLE', 'NO_RUMPLE']"
ADAM_MAXITER = 50000
LBFGS_MAXITER = 15000

######################################################################
USE_EQN = False
USE_MATCHING  = False 

######################################################################
if TEST_CASE == 'NO_RUMPLE':
    # EMBEDDING_CONFIG = [
    #     dict(embedding=True, embed_n=64, embed_std=3.0),
    #     dict(embedding=True, embed_n=32, embed_std=1.0)
    # ]
    EMBEDDING_CONFIG=None
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/norumple_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{'match_' if USE_MATCHING else''}{'eqn_' if USE_EQN else ''}no_rumple_checkpoints")
elif TEST_CASE == 'RUMPLE':
    EMBEDDING_CONFIG = [
        dict(embedding=True, embed_n=32, embed_std=2.0),
        dict(embedding=True, embed_n=32, embed_std=2.0),
        dict(embedding=True, embed_n=32, embed_std=2.0)
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{'match_' if USE_MATCHING else''}{'eqn_' if USE_EQN else ''}checkpoints")

os.makedirs(CKPT_PATH, exist_ok=True)

checkpointer = ocp.StandardCheckpointer() 

##########################################################################
############################# DATA PREPROCESSING #########################
##########################################################################
class DataOutput(NamedTuple):
    data_all: Any
    basal_mask: List[bool]
    idxgall: List[int]
    scale: Any
    
def load_data(data_path: str) -> DataOutput:
    print(f"[2] Loading and normalizing data from: {data_path}")
    rawdata = loadmat(data_path)

    basal_mask = [bool(b) for b in rawdata['basal_mask'].flatten()]
    print(f" . . . basal_mask (all regions): {basal_mask}")
    REGION_INDICES = list(range(len(basal_mask)))

    cell_keys = ['xd', 'yd', 'ud', 'vd', 
                'xd_h', 'yd_h', 'hd', 'sd', 
                'xct', 'yct', 'nnct',
                'xdir', 'ydir', 'udir', 'vdir',
                'xcol', 'ycol']
    for key in cell_keys:
        if key in rawdata and rawdata[key].dtype == object:
            rawdata[key] = rawdata[key][:, REGION_INDICES]

    idxcrop_orig = rawdata['idxcrop']
    idxcrop_h_orig = rawdata['idxcrop_h']

    data_all, idxgall, posi_all, idxcrop_all = normdata_xpinn(rawdata, basal_mask=basal_mask, use_regression=True)
    scale:List[SubScaleResult] = tree_map(lambda x: data_all[x][4][6], idxgall)
    return DataOutput(data_all, basal_mask, idxgall, scale)


##########################################################################
########################### XPINN INITIALIZATION #########################
##########################################################################
class XPINNOutput(NamedTuple):
    params: ArrayLike 
    sol_NN: Tuple[Callable]
    eval_f: Callable 
    
def initialize_xpinn(keys, data_output: DataOutput) -> XPINNOutput:
    _, basal_mask, idxgall, scale, = data_output
    n_hl = 6
    n_unit = 30
    
    print(data_output.idxgall)
    print(EMBEDDING_CONFIG)
    subkey, keys = random.split(keys, 2)
    params = init_xpinn(
        subkey, n_hl, n_unit, n_sub=len(idxgall), basal_mask=basal_mask, embedding_config=EMBEDDING_CONFIG)
    pred_u, grad_u = solu_xpinn(scale, basal_mask=basal_mask)
    sol_NN = (pred_u, grad_u)
    print(len(scale))
    eval_f = lambda params, x, idx: ssa_iso(lambda z: pred_u(params, z, idx), x, scale[idx], basal=basal_mask[idx])

    return keys, XPINNOutput(params, sol_NN, eval_f)


##########################################################################
######################### LOSS FUNC INITIALIZATION #######################
##########################################################################
class LossOutput(NamedTuple):
    data_f: Callable
    data_f_lbfgs: Callable
    loss_f: Callable 
    initial_loss: float
    sol_f: Callable


class ResumeState(NamedTuple):
    step: int
    params: ArrayLike
    opt_state: Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    return parser.parse_args()


def load_adam_checkpoint(resume_checkpoint: str | None) -> ResumeState | None:
    if resume_checkpoint is None:
        return None

    ckpt_path = os.path.abspath(resume_checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if os.path.commonpath([REGRESSION_ROOT, ckpt_path]) != REGRESSION_ROOT:
        raise ValueError(f"Checkpoint must be inside {REGRESSION_ROOT}: {ckpt_path}")

    ckpt_name = os.path.basename(ckpt_path)
    if ckpt_name.startswith('LBFGS_'):
        raise ValueError(f"L-BFGS checkpoints are not supported for resume: {ckpt_name}")
    if re.fullmatch(r'step_\d+\.pkl', ckpt_name) is None:
        raise ValueError(f"Resume checkpoint must match step_[num].pkl: {ckpt_name}")

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    required_keys = {'step', 'params', 'opt_state'}
    missing_keys = required_keys.difference(ckpt)
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {', '.join(sorted(missing_keys))}")

    step = int(ckpt['step'])
    print(f"[resume] Loaded ADAM checkpoint: {ckpt_path} (step {step})")
    return ResumeState(
        step=step,
        params=jax.device_put(ckpt['params']),
        opt_state=jax.device_put(ckpt['opt_state'])
    )
    
def initialize_loss(keys, data_output: DataOutput, xpinn_output: XPINNOutput) -> LossOutput:
    data_all, basal_mask, idxgall, scale = data_output 
    params, sol_NN, eval_f = xpinn_output 
    
    lw = [1.0, 0.0, 0.0, 0.0]
    if len(idxgall) == 3:
        n_pt = [
            [2500, 2500, 200],
            [2500, 2500, 200]
        ]
    elif len(idxgall) == 2:
        n_pt = [
            [2500, 2500],
            [2500, 2500]
        ]

    n_pt_lbfgs = [[2 * n for n in group] for group in n_pt]
        
    data_f = dsample_regression_xpinn(data_all, idxgall, n_pt)
    data_f_lbfgs = dsample_regression_xpinn(data_all, idxgall, n_pt_lbfgs)
    
    dkey, keys = random.split(keys, 2)
    smp = data_f(dkey)
    
    NN_loss = loss_regression_xpinn(sol_NN, idxgall, basal_mask=basal_mask, eqn=ssa_iso if USE_EQN else None, match=USE_MATCHING, scales=data_output.scale)
        
    initial_loss = NN_loss(params, smp)[0]
    NN_loss.lref = initial_loss
    return keys, LossOutput(data_f, data_f_lbfgs, NN_loss, initial_loss, sol_NN)

def optimize(keys, data: DataOutput, xpinn: XPINNOutput, loss: LossOutput, resume_state: ResumeState | None = None):
    start_learning_rate = 1e-3
    
    optimizer = optax.adam(start_learning_rate)
    if resume_state is None:
        params = xpinn.params
        opt_state = optimizer.init(params)
        adam_start_step = 0
    else:
        params = resume_state.params
        opt_state = resume_state.opt_state
        adam_start_step = resume_state.step + 1
    
    calc_loss = lambda params, x: loss.loss_f(params, x)[0]
    
    def save_checkpoint(step, params, opt_state, is_lbfgs=False):
        if not is_lbfgs:
            ckpt = {
                'step': step,
                'params': jax.device_get(params),
                'opt_state': jax.device_get(opt_state)
            }
        else:
            ckpt = {
                'step': step,
                'params': jax.device_get(params)
            }
            
        with open(os.path.join(CKPT_PATH, f"{'LBFGS_' if is_lbfgs else ''}step_{step}.pkl"), 'wb') as f:
            pickle.dump(ckpt, f)


    @jax.jit
    def train_step(params, opt_state, smp):
        l, grads = jax.value_and_grad(calc_loss)(params, smp)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, l

    lbfgs_solver = optax.lbfgs(
        memory_size=10,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            initial_guess_strategy='one'
        )
    )
    lbfgs_value_and_grad = optax.value_and_grad_from_state(calc_loss)

    @jax.jit
    def lbfgs_step(params, opt_state, smp):
        value, grad = lbfgs_value_and_grad(params, state=opt_state, x=smp)
        updates, new_opt_state = lbfgs_solver.update(
            grad, opt_state, params,
            value=value, grad=grad,
            value_fn=calc_loss,
            x=smp
        )
        new_params = optax.apply_updates(params, updates)
        grad_inf = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grad)]))
        return new_params, new_opt_state, value, grad_inf
    
    def print_subregion_errs(reg_err_list):
        data_err = reg_err_list[0]
        eqn_err = reg_err_list[1]
        md_err = reg_err_list[2]
        n_region = len(data.idxgall)
        for idx in range(len(data_err)):
            subregion_data_errs = data_err[idx]
            subregion_eqn_errs = eqn_err[idx]
            if idx==0:
                subregion_md_errs = md_err[idx]
            elif idx==n_region-1:
                subregion_md_errs = md_err[idx-1]
            else:
                subregion_md_errs = md_err[idx]

            print(f'                  Region {idx}: u = {subregion_data_errs[0]:<5.3e} | v = {subregion_data_errs[1]:<5.3e} | h = {subregion_data_errs[2]:<5.3e} | s = {subregion_data_errs[3]:<5.3e} | mu= {subregion_data_errs[4]:<5.3e} | c = {subregion_data_errs[5]:<5.3e}')
            print(f'                            eqn: x = {subregion_eqn_errs[0]:<5.3e} | y = {subregion_eqn_errs[1]:<5.3e}')
            print(f'                            md:  u = {subregion_md_errs[0]:<5.3e} | v = {subregion_md_errs[1]:<5.3e} | h = {subregion_md_errs[2]:<5.3e} | s = {subregion_md_errs[3]:<5.3e} | mu = {subregion_md_errs[4]:<5.3e}')

    
    loss_history = []
    
    if adam_start_step >= ADAM_MAXITER:
        print(f'[resume] ADAM checkpoint step {adam_start_step - 1} already reaches ADAM_MAXITER={ADAM_MAXITER}; starting L-BFGS directly.')
    else:
        for step in range(adam_start_step, ADAM_MAXITER):
            dkey, keys = random.split(keys, 2)
            smp = loss.data_f(dkey)
            params, opt_state, l = train_step(params, opt_state, smp)
            loss_history.append(l)
            
            if step % 100 == 0:
                _, loss_info, reg_err_list = loss.loss_f(params, smp)
                print(f'--------------------------------- STEP {step} ------------------------------------')
                print(f'ADAM step {step}: loss={loss_info[0]:.2e} | u={loss_info[1]:.2e} | v={loss_info[2]:.2e} | h={loss_info[3]:.2e} | s={loss_info[4]:.2e} | mu={loss_info[5]:.2e} | C={loss_info[6]:.2e}')
                print(f'                  eqnx={loss_info[7]:.2e} | eqny={loss_info[8]:.2e}')
                print_subregion_errs(reg_err_list)
            if step % 1000 == 0:
                save_checkpoint(step, params, opt_state)

    lbfgs_key, keys = random.split(keys, 2)
    data_lbfgs = loss.data_f_lbfgs(lbfgs_key)
    loss_before_lbfgs = calc_loss(params, data_lbfgs)
    
    print('--------------------------------- L-BFGS ------------------------------------')
    print(f'L-BFGS start: loss={loss_before_lbfgs:.2e}')
    lbfgs_state = lbfgs_solver.init(params)
    lbfgs_reason = 'max_iterations'
    lbfgs_iters = 0
    loss_after_lbfgs = loss_before_lbfgs

    for lbfgs_step_idx in range(LBFGS_MAXITER):
        params, lbfgs_state, loss_value, grad_inf = lbfgs_step(params, lbfgs_state, data_lbfgs)
        loss_value = float(loss_value)
        grad_inf = float(grad_inf)
        loss_history.append(loss_value)
        lbfgs_iters = lbfgs_step_idx + 1

        if lbfgs_iters <= 10 or lbfgs_iters % LBFGS_LOG_RATE == 0:
            print(f'L-BFGS step {lbfgs_iters}: loss={loss_value:.2e} | grad_inf={grad_inf:.2e}')

        if not jnp.isfinite(loss_value):
            lbfgs_reason = 'non_finite_loss'
            loss_after_lbfgs = loss_value
            break
        if grad_inf < LBFGS_TOL:
            lbfgs_reason = 'grad_tol'
            loss_after_lbfgs = loss_value
            break

        loss_after_lbfgs = loss_value

    print(f'L-BFGS end: loss={loss_after_lbfgs:.2e}')
    print(f'             reason={lbfgs_reason} | iterations={lbfgs_iters}')
    save_checkpoint(lbfgs_iters, params, None, is_lbfgs=True)

    return params, loss_history
    
if __name__ == '__main__': 
    args = parse_args()
    keys = random.PRNGKey(8132002)
    
    data_output = load_data(DATA_PATH) 
    keys, xpinn_output = initialize_xpinn(keys, data_output)
    resume_state = load_adam_checkpoint(args.resume_checkpoint)
    if resume_state is not None:
        xpinn_output = XPINNOutput(resume_state.params, xpinn_output.sol_NN, xpinn_output.eval_f)
    keys, loss_output = initialize_loss(keys, data_output, xpinn_output)
    print(f'Initial X-PINN Loss = {loss_output.initial_loss:.2e}')
    params, loss_history = optimize(keys, data_output, xpinn_output, loss_output, resume_state=resume_state)
    with open(os.path.join(CKPT_PATH, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(loss_history, f)
