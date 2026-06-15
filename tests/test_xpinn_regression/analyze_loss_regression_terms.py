import argparse
import json
import os

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_leaves

from tests.test_xpinn_regression import xpinn_regression as xr


DATA_TERM_NAMES = ('data_u', 'data_v', 'data_h', 'data_s', 'data_log_mu', 'data_C')
EQN_TERM_NAMES = ('eqn_x', 'eqn_y')
MATCH_TERM_NAMES = (
    'match_C0_u', 'match_C0_v', 'match_C0_h', 'match_C0_s', 'match_C0_log_mu',
    'match_C1_u_x', 'match_C1_u_y', 'match_C1_v_x', 'match_C1_v_y',
    'match_C1_h_x', 'match_C1_h_y', 'match_C1_log_mu_x', 'match_C1_log_mu_y',
)
CT_TERM_NAMES = ('ct_x', 'ct_y')
GPINN_TERM_NAMES = ('gpinn_eqn_x_x', 'gpinn_eqn_x_y', 'gpinn_eqn_y_x', 'gpinn_eqn_y_y')
MU_GRAD_TERM_NAMES = ('mu_grad_x', 'mu_grad_y')


def tree_l2_norm(tree):
    leaves = [jnp.ravel(x) for x in tree_leaves(tree) if x is not None and x.size > 0]
    if len(leaves) == 0:
        return 0.0
    return float(jnp.linalg.norm(jnp.concatenate(leaves)))


def tree_linf_norm(tree):
    leaves = [jnp.max(jnp.abs(x)) for x in tree_leaves(tree) if x is not None and x.size > 0]
    if len(leaves) == 0:
        return 0.0
    return float(jnp.max(jnp.stack(leaves)))


def load_analysis_context(checkpoint_path, batch_seed):
    key = random.PRNGKey(42)
    data_output = xr.load_data(xr.DATA_PATH)
    key, xpinn_output = xr.initialize_xpinn(key, data_output)
    resume = xr.load_checkpoint(checkpoint_path)
    params = resume.params

    key, loss_output = xr.initialize_loss(key, data_output, xpinn_output)
    batch = loss_output.data_f(random.PRNGKey(batch_seed))
    batch = xr.attach_loss_weights(batch, int(resume.step), data_output.idxgall)

    return {
        'data_output': data_output,
        'resume': resume,
        'params': params,
        'loss_fn': loss_output.loss_f,
        'batch': batch,
        'match_weight': float(xr.matching_weight(int(resume.step))),
        'eqn_region_weights': [float(x) for x in batch['eqn_region_weights']],
        'data_w': jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        'eqn_w': jnp.array([1.0, 1.0]),
        'md_w': jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
        'ct_w': jnp.array([30.0, 30.0]),
    }


def evaluate_components(loss_fn, params, batch):
    _, loss_info, reg_err_list = loss_fn(params, batch)
    data_err_list, eqn_err_list, md_err_list, ct_err_list = reg_err_list
    eqn_region_weights = batch['eqn_region_weights'].reshape((-1, 1))

    return {
        'total_loss': loss_info[0],
        'data_terms': jnp.mean(jnp.array(data_err_list), axis=0),
        'eqn_terms': jnp.mean(eqn_region_weights * jnp.array(eqn_err_list), axis=0),
        'match_terms': jnp.mean(jnp.array(md_err_list), axis=0),
        'ct_terms': jnp.mean(jnp.array(ct_err_list), axis=0),
        'gpinn_mean': loss_info[xr.GPINN_LOSS_INFO_IDX],
        'gpinn_terms': loss_info[xr.GPINN_COMPONENT_INFO_SLICE],
        'mu_grad_mean': loss_info[xr.MU_GRAD_LOSS_INFO_IDX],
        'mu_grad_terms': loss_info[xr.MU_GRAD_COMPONENT_INFO_SLICE],
    }


def build_term_specs(ctx):
    match_weight = ctx['match_weight']
    data_w = ctx['data_w']
    eqn_w = ctx['eqn_w']
    md_w = ctx['md_w']
    ct_w = ctx['ct_w']
    gpinn_weight = float(ctx['loss_fn'].gpinn_weight)
    mu_grad_weight = float(ctx['loss_fn'].mu_grad_weight)

    specs = []

    for i, name in enumerate(DATA_TERM_NAMES):
        specs.append({
            'category': 'data',
            'name': name,
            'scale': float(data_w[i] / len(DATA_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['data_terms'][i],
        })

    for i, name in enumerate(EQN_TERM_NAMES):
        specs.append({
            'category': 'eqn',
            'name': name,
            'scale': float(0.01 * eqn_w[i] / len(EQN_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['eqn_terms'][i],
        })

    for i, name in enumerate(MATCH_TERM_NAMES):
        specs.append({
            'category': 'match',
            'name': name,
            'scale': float(match_weight * md_w[i] / len(MATCH_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['match_terms'][i],
        })

    for i, name in enumerate(CT_TERM_NAMES):
        specs.append({
            'category': 'ct',
            'name': name,
            'scale': float(0.01 * ct_w[i] / len(CT_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['ct_terms'][i],
        })

    for i, name in enumerate(GPINN_TERM_NAMES):
        specs.append({
            'category': 'gpinn',
            'name': name,
            'scale': float(gpinn_weight / len(GPINN_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['gpinn_terms'][i],
        })

    for i, name in enumerate(MU_GRAD_TERM_NAMES):
        specs.append({
            'category': 'mu_grad',
            'name': name,
            'scale': float(mu_grad_weight / len(MU_GRAD_TERM_NAMES)),
            'getter': lambda comp, i=i: comp['mu_grad_terms'][i],
        })

    return specs


def analyze_terms(ctx):
    loss_fn = ctx['loss_fn']
    params = ctx['params']
    batch = ctx['batch']
    components = evaluate_components(loss_fn, params, batch)
    term_specs = build_term_specs(ctx)

    rows = []
    for spec in term_specs:
        raw_value = float(spec['getter'](components))
        direct_contribution = spec['scale'] * raw_value

        def scalar_term(p, spec=spec):
            comp = evaluate_components(loss_fn, p, batch)
            return spec['scale'] * spec['getter'](comp)

        grad = jax.grad(scalar_term)(params)
        rows.append({
            'category': spec['category'],
            'name': spec['name'],
            'raw_value': raw_value,
            'scale': spec['scale'],
            'direct_contribution': float(direct_contribution),
            'grad_l2': tree_l2_norm(grad),
            'grad_linf': tree_linf_norm(grad),
        })

    def group_sum(prefix):
        return float(sum(row['direct_contribution'] for row in rows if row['category'] == prefix))

    summaries = {
        'total_loss': float(components['total_loss']),
        'data_contribution': group_sum('data'),
        'eqn_contribution': group_sum('eqn'),
        'match_contribution': group_sum('match'),
        'ct_contribution': group_sum('ct'),
        'gpinn_contribution': group_sum('gpinn'),
        'mu_grad_contribution': group_sum('mu_grad'),
        'gpinn_mean_raw': float(components['gpinn_mean']),
        'mu_grad_mean_raw': float(components['mu_grad_mean']),
    }

    def total_loss_scalar(p):
        return loss_fn(p, batch)[1][0]

    total_grad = jax.grad(total_loss_scalar)(params)
    summaries['total_grad_l2'] = tree_l2_norm(total_grad)
    summaries['total_grad_linf'] = tree_linf_norm(total_grad)

    return rows, summaries


def default_checkpoint():
    return os.path.join(
        os.path.dirname(__file__),
        'match_ct_eqn_subglacial_channel_checkpoints',
        'KFAC_step_4000.pkl',
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=default_checkpoint())
    parser.add_argument('--batch-seed', type=int, default=0)
    parser.add_argument('--json', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    ctx = load_analysis_context(os.path.abspath(args.checkpoint), args.batch_seed)
    rows, summaries = analyze_terms(ctx)

    output = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'step': int(ctx['resume'].step),
        'match_weight': ctx['match_weight'],
        'eqn_region_weights': ctx['eqn_region_weights'],
        'batch_seed': args.batch_seed,
        'summaries': summaries,
        'terms': rows,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"checkpoint={output['checkpoint']}")
    print(f"step={output['step']}")
    print(f"batch_seed={output['batch_seed']}")
    print(f"match_weight={output['match_weight']:.12e}")
    print(f"eqn_region_weights={output['eqn_region_weights']}")
    print('SUMMARIES')
    for key, value in summaries.items():
        print(f'{key}={value:.12e}')
    print('TERMS')
    for row in rows:
        print('\t'.join([
            row['category'],
            row['name'],
            f"raw_value={row['raw_value']:.12e}",
            f"scale={row['scale']:.12e}",
            f"direct_contribution={row['direct_contribution']:.12e}",
            f"grad_l2={row['grad_l2']:.12e}",
            f"grad_linf={row['grad_linf']:.12e}",
        ]))


if __name__ == '__main__':
    main()
