import jax.numpy as jnp
import pickle
import jax
from jax import config

#[NOTE]: Double-precision floating point arithmetic can be importantwhen equation loss is small.
config.update("jax_enable_x64", True)
IS_METAL_BACKEND = jax.default_backend().lower() == 'metal'
if IS_METAL_BACKEND:
    config.update("jax_enable_x64", False)

import random as pyrnd
from jax import random
from jax.tree_util import tree_map, tree_leaves
from scipy.io import loadmat
import time, os, pickle, argparse
import re
import optax

from diffice_jax import normdata_xpinn, dsample_regression_xpinn
import diffice_jax
print(f"[1] DIFFICE_jax path at {diffice_jax.__file__}")

from diffice_jax import ssa_iso, dbc_iso
from diffice_jax import init_xpinn, solu_xpinn
from diffice_jax import loss_iso_xpinn
from diffice_jax import adam_opt
from diffice_jax import loss_regression_xpinn

from diffice_jax.data.xpinns.preprocessing import SubScaleResult

from datetime import datetime
import json, copy
from typing import NamedTuple, Any, List, Tuple, Callable
from jax.typing import ArrayLike

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REGRESSION_ROOT = os.path.join(PROJECT_ROOT, 'test_xpinn_regression')

#######################################################################
TEST_CASE = 'FLATBED'
assert TEST_CASE in ['RUMPLE', 'NO_RUMPLE', 'NO_RUMPLE_SPARSETHK', 'SUBGLACIAL_CHANNEL', 'FLATBED'], "TEST_CASE must be one of ['RUMPLE', 'NO_RUMPLE', 'NO_RUMPLE_SPARSETHK', 'SUBGLACIAL_CHANNEL', 'FLATBED']"
ADAM_MAXITER = 60000
KFAC_MAXITER = 100000

KFAC_LOG_RATE = 100
KFAC_CKPT_RATE = 1000

OPTIMIZER = 'KFAC'

MATCH_WEIGHT_START = 1.0
MATCH_WEIGHT_STOP = 1.0
MATCH_WEIGHT_FACTOR = 1.004

EQN_WEIGHT_REGIONS = [0, 1]
EQN_WEIGHT_START = 1. # This weight multiplies whatever weight was assigned to the equation already
EQN_WEIGHT_STOP = 1
EQN_WEIGHT_FACTOR = 1.0004

USE_GPINN = True  
GPINN_WEIGHT = 0.001
USE_GPINN_IN_KFAC = True

USE_MU_GRAD_REG = False
MU_GRAD_WEIGHT = 1e-2
USE_MU_GRAD_IN_KFAC = False

USE_REGION_TERM_BALANCING = False

######################################################################
USE_EQN = True 
USE_MATCHING  = True
USE_CT = True
TRAIN_MODE = 'full'
TRAIN_MODE_CHOICES = ('full', 'grounded_only_interface_mu_ct', 'floating_region1_only', 'grounded_region0_from_floating')
assert TRAIN_MODE in TRAIN_MODE_CHOICES
USE_GROUNDED_ONLY_INTERFACE_MU_CT = TRAIN_MODE == 'grounded_only_interface_mu_ct'
USE_FLOATING_REGION1_ONLY = TRAIN_MODE == 'floating_region1_only'
USE_GROUNDED_REGION0_FROM_FLOATING = TRAIN_MODE == 'grounded_region0_from_floating'
assert not (USE_GROUNDED_ONLY_INTERFACE_MU_CT and USE_FLOATING_REGION1_ONLY)

NETWORK_FAMILIES = ('net_u', 'net_mu', 'net_c0', 'net_c1', 'net_u1')


def freeze_regions_spec(regions):
    return {family: list(regions) for family in NETWORK_FAMILIES}


def kfac_loss_terms():
    return tuple(
        term for term, enabled in (
            ('eqn', USE_EQN),
            ('ct', EFFECTIVE_USE_CT),
            ('match', EFFECTIVE_USE_MATCHING),
            ('gpinn', USE_GPINN and USE_GPINN_IN_KFAC),
            ('mu_grad', USE_MU_GRAD_REG and USE_MU_GRAD_IN_KFAC),
        )
        if enabled
    )


def set_train_mode(mode):
    global TRAIN_MODE, USE_GROUNDED_ONLY_INTERFACE_MU_CT, USE_FLOATING_REGION1_ONLY
    global USE_GROUNDED_REGION0_FROM_FLOATING, EFFECTIVE_USE_CT, EFFECTIVE_USE_MATCHING
    global TRAIN_BRANCH_PREFIX, KFAC_LOSS_TERMS, KFAC_PHYS_OBJECTIVE_WEIGHT
    TRAIN_MODE = mode
    USE_GROUNDED_ONLY_INTERFACE_MU_CT = mode == 'grounded_only_interface_mu_ct'
    USE_FLOATING_REGION1_ONLY = mode == 'floating_region1_only'
    USE_GROUNDED_REGION0_FROM_FLOATING = mode == 'grounded_region0_from_floating'
    EFFECTIVE_USE_CT = USE_CT and not USE_GROUNDED_REGION0_FROM_FLOATING
    EFFECTIVE_USE_MATCHING = USE_MATCHING and not USE_GROUNDED_ONLY_INTERFACE_MU_CT and not USE_FLOATING_REGION1_ONLY
    TRAIN_BRANCH_PREFIX = (
        'grounded_only_interface_mu_ct_' if USE_GROUNDED_ONLY_INTERFACE_MU_CT else
        'floating_region1_only_' if USE_FLOATING_REGION1_ONLY else
        'grounded_region0_from_floating_' if USE_GROUNDED_REGION0_FROM_FLOATING else
        ''
    )
    KFAC_LOSS_TERMS = kfac_loss_terms()
    KFAC_PHYS_OBJECTIVE_WEIGHT = 100.0 if USE_FLOATING_REGION1_ONLY else 1.0


ACTIVE_REGIONS = [1] if (USE_GROUNDED_ONLY_INTERFACE_MU_CT or USE_FLOATING_REGION1_ONLY) else [0, 1]
FLOATING_REGIONS = [0] if USE_GROUNDED_ONLY_INTERFACE_MU_CT else ([1] if USE_FLOATING_REGION1_ONLY else [])
ACTIVE_INTERFACES = [] if USE_GROUNDED_ONLY_INTERFACE_MU_CT else None
EFFECTIVE_USE_CT = USE_CT and not USE_GROUNDED_REGION0_FROM_FLOATING
CT_COMPONENT_COUNT = 1 if USE_GROUNDED_ONLY_INTERFACE_MU_CT else 2
EFFECTIVE_USE_MATCHING = USE_MATCHING and not USE_GROUNDED_ONLY_INTERFACE_MU_CT and not USE_FLOATING_REGION1_ONLY
TRAIN_BRANCH_PREFIX = (
    'grounded_only_interface_mu_ct_' if USE_GROUNDED_ONLY_INTERFACE_MU_CT else
    'floating_region1_only_' if USE_FLOATING_REGION1_ONLY else
    'grounded_region0_from_floating_' if USE_GROUNDED_REGION0_FROM_FLOATING else
    ''
)

USE_ADAPTIVE_SAMPLING = True
ADAPT_PERIOD = 50
ADAPT_BURNIN = 1000
REGRESSION_N_PT_BY_NSUB = {
    3: [
        [2500, 2500, 200],
        [2500, 2500, 200],
        [2500, 2500, 200],
    ],
    2: [
        [1000, 1000],  # Velocity 
        [1000, 1000], # Thickness 
        [1000, 1000],   # Collocation
    ],
}

# The active branch can freeze complete subregion networks while training the rest.
# FREEZE_NETWORKS = dict(
#     net_u=[0, 1],
#     net_mu=[1],
#     net_c0=[0, 1],
# )
FREEZE_NETWORKS = (
    freeze_regions_spec([0])
    if USE_GROUNDED_ONLY_INTERFACE_MU_CT or USE_FLOATING_REGION1_ONLY else
    dict(net_u=[], net_mu=[], net_c0=[])
)
KFAC_LOSS_TERMS = kfac_loss_terms()
KFAC_ACTIVE_REGIONS = ACTIVE_REGIONS
USE_DATA_IN_KFAC = True
KFAC_DATA_REGIONS = ACTIVE_REGIONS
KFAC_PHYS_OBJECTIVE_WEIGHT = 100.0 if USE_FLOATING_REGION1_ONLY else 1.0
KFAC_DATA_OBJECTIVE_WEIGHT = 1.0
MATCH_COMPONENT_WEIGHTS_ON = (1., 1., 1., 1., 1.,
                              0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0,
                              0.06, 0.06, 0.06, 0.06, 0.06, 0.06)
######################################################################

LOSS_INFO_TOTAL_IDX = 0
LOSS_INFO_DATA_SLICE = slice(1, 7)
LOSS_INFO_EQN_SLICE = slice(7, 9)
MATCH_COMPONENT_COUNT = 19
LOSS_INFO_MATCH_SLICE = slice(9, 9 + MATCH_COMPONENT_COUNT)
LOSS_INFO_CT_SLICE = slice(LOSS_INFO_MATCH_SLICE.stop, LOSS_INFO_MATCH_SLICE.stop + CT_COMPONENT_COUNT)
GPINN_LOSS_INFO_IDX = LOSS_INFO_CT_SLICE.stop
GPINN_COMPONENT_INFO_SLICE = slice(GPINN_LOSS_INFO_IDX + 1, GPINN_LOSS_INFO_IDX + 5)
MU_GRAD_LOSS_INFO_IDX = GPINN_COMPONENT_INFO_SLICE.stop
MU_GRAD_COMPONENT_INFO_SLICE = slice(MU_GRAD_LOSS_INFO_IDX + 1, MU_GRAD_LOSS_INFO_IDX + 3)
REGION_TERM_ORDER = (
    ('data', 'eqn', 'ct', 'match', 'gpinn', 'mu_grad')
    if USE_MU_GRAD_REG else
    ('data', 'eqn', 'ct', 'match', 'gpinn')
)
REGION_TERM_UPDATE_PERIOD = 100


def matching_weight(step):
    if not USE_MATCHING or USE_GROUNDED_ONLY_INTERFACE_MU_CT:
        return jnp.array(1.0)
    return jnp.minimum(MATCH_WEIGHT_START * MATCH_WEIGHT_FACTOR ** step, MATCH_WEIGHT_STOP)


def eqn_region_weight(step):
    return jnp.minimum(EQN_WEIGHT_START * EQN_WEIGHT_FACTOR ** step, EQN_WEIGHT_STOP)


def eqn_region_weights(step, idxgall):
    weights = jnp.ones(len(idxgall))
    if len(EQN_WEIGHT_REGIONS) == 0:
        return weights
    weight = eqn_region_weight(step)
    for idx in EQN_WEIGHT_REGIONS:
        weights = weights.at[idxgall.index(idx)].set(weight)
    return weights


def attach_loss_weights(batch, step, idxgall):
    batch = dict(batch)
    batch['match_weight'] = matching_weight(step)
    batch['eqn_region_weights'] = eqn_region_weights(step, idxgall)
    batch['active_regions'] = ACTIVE_REGIONS
    return batch


def active_region_terms(basal_mask, idx_pos, n_sub):
    if idx_pos not in ACTIVE_REGIONS:
        return tuple()
    is_basal = basal_mask[idx_pos]
    active = ['data']
    if USE_EQN:
        active.append('eqn')
    if EFFECTIVE_USE_CT and ((not is_basal) or USE_GROUNDED_ONLY_INTERFACE_MU_CT):
        active.append('ct')
    if USE_GPINN:
        active.append('gpinn')
    if USE_MU_GRAD_REG and not is_basal:
        active.append('mu_grad')
    if EFFECTIVE_USE_MATCHING and (idx_pos > 0 or idx_pos < n_sub - 1):
        active.append('match')
    return tuple(active)


def initialize_region_term_weights(idxgall, basal_mask):
    n_sub = len(idxgall)
    weights = {
        term: jnp.zeros((n_sub,), dtype=jnp.float64)
        for term in REGION_TERM_ORDER
    }
    for idx_pos in range(n_sub):
        active = active_region_terms(basal_mask, idx_pos, n_sub)
        if len(active) == 0:
            continue
        value = 1.0 / len(active)
        for term in active:
            weights[term] = weights[term].at[idx_pos].set(value)
    return weights


def tree_l2_norm(tree):
    leaves = [jnp.ravel(x) for x in tree_leaves(tree) if x is not None and x.size > 0]
    if len(leaves) == 0:
        return 0.0
    return float(jnp.linalg.norm(jnp.concatenate(leaves)))


def compute_region_term_weights(params, batch, loss_f, idxgall, basal_mask, frozen_params, freeze_mask, previous_weights):
    n_sub = len(idxgall)
    weights = {term: jnp.array(previous_weights[term]) for term in REGION_TERM_ORDER}

    for idx_pos in range(n_sub):
        active = active_region_terms(basal_mask, idx_pos, n_sub)
        if len(active) == 0:
            continue

        grad_norms = {}
        for term in active:
            def term_value(p, term=term, idx_pos=idx_pos):
                active_params = train_view_params(p, frozen_params, freeze_mask)
                return loss_f.region_term_values(active_params, batch)[term][idx_pos]

            grad_norms[term] = tree_l2_norm(jax.grad(term_value)(params))

        norm_sum = sum(grad_norms.values())
        if norm_sum > 0.0 and all(grad_norms[term] > 0.0 for term in active):
            for term in REGION_TERM_ORDER:
                value = grad_norms[term] / norm_sum if term in grad_norms else 0.0
                weights[term] = weights[term].at[idx_pos].set(value)

    return weights


def format_region_term_weights(batch, data_output):
    if not USE_REGION_TERM_BALANCING:
        return 'region_term_w=off'
    if 'region_term_weights' not in batch:
        return 'region_term_w=off'

    lines = []
    weights = batch['region_term_weights']
    for idx_pos, idx in enumerate(data_output.idxgall):
        active = active_region_terms(data_output.basal_mask, idx_pos, len(data_output.idxgall))
        items = ', '.join(f'{term}={float(weights[term][idx_pos]):.2e}' for term in active)
        lines.append(f'region {idx}: {items}')
    return 'region_term_w=[' + ' | '.join(lines) + ']'


def net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)):
    return dict(
        net_u=dict(depth=u[0], width=u[1]),
        net_mu=dict(depth=mu[0], width=mu[1]),
        net_c0=dict(depth=c0[0], width=c0[1]),
        net_c1=dict(depth=c1[0], width=c1[1]),
    )


def format_eqn_errs(eqn_errs, is_basal):
    return f'x = {eqn_errs[0]:<5.3e} | y = {eqn_errs[1]:<5.3e}'


def format_ct_errs(ct_errs, is_basal):
    if not EFFECTIVE_USE_CT:
        return 'off'
    if USE_GROUNDED_ONLY_INTERFACE_MU_CT and is_basal:
        return f'log_mu = {ct_errs[0]:<5.3e}'
    return f'x = {ct_errs[0]:<5.3e} | y = {ct_errs[1]:<5.3e}'


def format_gpinn_loss(loss_info):
    if not USE_GPINN:
        return 'gpinn=off'
    gpinn_raw = loss_info[GPINN_LOSS_INFO_IDX]
    gpinn_weighted = GPINN_WEIGHT * gpinn_raw
    components = loss_info[GPINN_COMPONENT_INFO_SLICE]
    return (f'gpinn={gpinn_weighted:.2e} | gpinn_raw={gpinn_raw:.2e} | '
            f'gpinn_dx=({components[0]:.2e}, {components[2]:.2e}) | '
            f'gpinn_dy=({components[1]:.2e}, {components[3]:.2e})')


def format_mu_grad_loss(loss_info):
    if not USE_MU_GRAD_REG:
        return 'mu_grad=off'
    mu_grad_raw = loss_info[MU_GRAD_LOSS_INFO_IDX]
    mu_grad_weighted = MU_GRAD_WEIGHT * mu_grad_raw if USE_MU_GRAD_REG else 0.0
    components = loss_info[MU_GRAD_COMPONENT_INFO_SLICE]
    return (f'mu_grad={mu_grad_weighted:.2e} | mu_grad_raw={mu_grad_raw:.2e} | '
            f'mu_grad_x={components[0]:.2e} | mu_grad_y={components[1]:.2e}')


def format_eqn_region_weights(batch):
    weights = [f'{float(w):.2e}' for w in batch['eqn_region_weights']]
    return f'eqn_region_w=[{", ".join(weights)}]'


def format_data_loss_summary(loss_info, u_label, v_label):
    data_terms = loss_info[LOSS_INFO_DATA_SLICE]
    return (f'{u_label}={data_terms[0]:.2e} | {v_label}={data_terms[1]:.2e} | '
            f'h={data_terms[2]:.2e} | s={data_terms[3]:.2e} | '
            f'mu={data_terms[4]:.2e} | C={data_terms[5]:.2e}')


def format_eqn_loss_summary(loss_info):
    eqn_terms = loss_info[LOSS_INFO_EQN_SLICE]
    return f'eqn0={eqn_terms[0]:.2e} | eqn1={eqn_terms[1]:.2e}'


def format_match_errs(md_errs):
    lines = [
        f'md:  u = {md_errs[0]:<5.3e} | v = {md_errs[1]:<5.3e} | h = {md_errs[2]:<5.3e} | s = {md_errs[3]:<5.3e} | mu = {md_errs[4]:<5.3e}',
        f'     ux = {md_errs[5]:<5.3e} | uy = {md_errs[6]:<5.3e} | vx = {md_errs[7]:<5.3e} | vy = {md_errs[8]:<5.3e} | hx = {md_errs[9]:<5.3e} | hy = {md_errs[10]:<5.3e}',
        f'     mux = {md_errs[11]:<5.3e} | muy = {md_errs[12]:<5.3e} | uxx = {md_errs[13]:<5.3e} | uxy = {md_errs[14]:<5.3e} | uyy = {md_errs[15]:<5.3e}',
        f'     vxx = {md_errs[16]:<5.3e} | vxy = {md_errs[17]:<5.3e} | vyy = {md_errs[18]:<5.3e}',
    ]
    return lines


def format_region_match_errs(idx, md_errs):
    if not EFFECTIVE_USE_MATCHING or idx not in ACTIVE_REGIONS:
        return ['md:  off']
    return format_match_errs(md_errs)


def concat_residuals(residuals):
    residuals = [r for r in residuals if r.shape[0] > 0]
    return jnp.concatenate(residuals, axis=0) if len(residuals) > 0 else jnp.zeros((0, 1))


def active_adaptive_eval(eval_f):
    def eval_active(x, idx, basal):
        if (USE_FLOATING_REGION1_ONLY or USE_GROUNDED_REGION0_FROM_FLOATING) and idx not in ACTIVE_REGIONS:
            return (jnp.zeros((x.shape[0], 2), dtype=x.dtype),)
        return eval_f(x, idx, basal)
    return eval_active


def _freeze_leaf(x, frozen):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _freeze_leaf(v, frozen) for k, v in x.items()}
    if isinstance(x, list):
        return [_freeze_leaf(v, frozen) for v in x]
    if isinstance(x, tuple):
        return tuple(_freeze_leaf(v, frozen) for v in x)
    return jnp.full_like(x, frozen, dtype=bool)


def normalize_freeze_spec(params, freeze_spec):
    normalized = {}
    for family, subnets in params.items():
        spec = freeze_spec.get(family, [])
        if spec == 'all':
            normalized[family] = [idx for idx, subnet in enumerate(subnets) if subnet is not None]
        else:
            normalized[family] = sorted(int(idx) for idx in spec)
    return normalized


def freeze_spec_changed(checkpoint_spec, current_spec):
    if checkpoint_spec is None:
        return any(len(v) > 0 for v in current_spec.values())
    checkpoint_spec = {k: sorted(v) for k, v in checkpoint_spec.items()}
    return checkpoint_spec != current_spec


def make_freeze_mask(params, freeze_spec):
    normalized = normalize_freeze_spec(params, freeze_spec)
    masks = {}
    for family, subnets in params.items():
        frozen_regions = set(normalized[family])
        masks[family] = [
            _freeze_leaf(subnet, idx in frozen_regions)
            for idx, subnet in enumerate(subnets)
        ]
    return masks, normalized


def apply_frozen_params(params, frozen_params, freeze_mask):
    def apply_leaf(param, frozen, mask):
        if mask is None:
            return param
        return jnp.where(mask, jax.lax.stop_gradient(frozen), param)
    return tree_map(apply_leaf, params, frozen_params, freeze_mask)


def project_frozen_params(params, frozen_params, freeze_mask):
    def project_leaf(param, frozen, mask):
        if mask is None:
            return param
        return jnp.where(mask, frozen, param)
    return tree_map(project_leaf, params, frozen_params, freeze_mask)


def train_view_params(params, frozen_params, freeze_mask):
    if USE_FLOATING_REGION1_ONLY:
        return params
    return apply_frozen_params(params, frozen_params, freeze_mask)


kfac_config = dict(
    # When learning_rate, momentum, and damping are each set to None
    # this enables the respective adaptive methods for them.
    # 1e-4, 0.9, and 1e-4 are sensible values, but the adaptive
    # methods tend to gives better results. Only the damping
    # adaptation method benefits from tuning
    learning_rate=None,
    momentum=None,
    damping=jnp.nan,
    norm_constraint=1e-8,  # ignored when LR or momentum adaptation is used
    initial_damping=1,  # used by the adaptive damping method
    min_damping=1e-6,  # used by the adaptive damping method
    curvature_block_type="naive_full",  # "naive_full" (very important setting)
    damping_adaptation_decay=0.997,
    curvature_ema=0.997,
    inverse_update_period=1,
    num_burnin_steps=0,  # TODO(jamesmartens): experiment with this
    always_use_exact_qmodel_for_damping_adjustment=True,
    include_norms_in_stats=True
    )


######################################################################
if TEST_CASE == 'NO_RUMPLE':
    # EMBEDDING_CONFIG = [
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=True, embed_n=32, embed_std=2.0),
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=False, embed_n=32, embed_std=2.0)
    # ]
    EMBEDDING_CONFIG = None
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/no_rumple_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}no_rumple_checkpoints")
elif TEST_CASE == 'FLATBED':
    # EMBEDDING_CONFIG = [
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=True, embed_n=32, embed_std=2.0),
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=False, embed_n=32, embed_std=2.0)
    # ]
    EMBEDDING_CONFIG = None
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/flatbed_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}flatbed_checkpoints")
elif TEST_CASE == 'MISMIP':
    # EMBEDDING_CONFIG = [
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=True, embed_n=32, embed_std=2.0),
    #     dict(embedding_u=False, embedding_mu=False, embedding_c=False, embed_n=32, embed_std=2.
    EMBEDDING_CONFIG = None
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/MISMIP_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}MISMIP_checkpoints")

elif TEST_CASE == 'SUBGLACIAL_CHANNEL':
    EMBEDDING_CONFIG = None
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/subglacial_channel_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}subglacial_channel_checkpoints")
elif TEST_CASE == 'RUMPLE':
    EMBEDDING_CONFIG = [
        dict(embedding_u=False, embedding_mu=True, embedding_c=True, embed_n=32, embed_std=2.0),
        dict(embedding_u=False, embedding_mu=True, embedding_c=True, embed_n=32, embed_std=2.0),
        dict(embedding_u=False, embedding_mu=True, embedding_c=True, embed_n=32, embed_std=2.0)
    ]
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}checkpoints")
elif TEST_CASE == 'NO_RUMPLE_SPARSETHK':
    EMBEDDING_CONFIG = [
        dict(embedding_u=False, embedding_mu=True, embedding_c=True, embed_n=64, embed_std=3.0),
        dict(embedding_u=False, embedding_mu=True, embedding_c=True, embed_n=32, embed_std=1.0)
    ]
    NETWORK_CONFIG = [
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)),
        net_arch(u=(6, 30), mu=(6, 30), c0=(6, 30), c1=(6, 30)),
    ]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'test_xpinn_regression/no_rumple_sparseThk_data_xpinns_regression_test.mat')
    CKPT_PATH = os.path.join(PROJECT_ROOT, f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}{'match_' if EFFECTIVE_USE_MATCHING else''}{'ct_' if EFFECTIVE_USE_CT else ''}{'eqn_' if USE_EQN else ''}no_rumple_sparseThk_checkpoints")

os.makedirs(CKPT_PATH, exist_ok=True)


def checkpoint_suffix():
    return dict(
        NO_RUMPLE='no_rumple_checkpoints',
        FLATBED='flatbed_checkpoints',
        MISMIP='MISMIP_checkpoints',
        SUBGLACIAL_CHANNEL='subglacial_channel_checkpoints',
        RUMPLE='checkpoints',
        NO_RUMPLE_SPARSETHK='no_rumple_sparseThk_checkpoints',
    )[TEST_CASE]


def refresh_checkpoint_path():
    global CKPT_PATH
    CKPT_PATH = os.path.join(
        PROJECT_ROOT,
        f"test_xpinn_regression/{TRAIN_BRANCH_PREFIX}"
        f"{'match_' if EFFECTIVE_USE_MATCHING else ''}"
        f"{'ct_' if EFFECTIVE_USE_CT else ''}"
        f"{'eqn_' if USE_EQN else ''}"
        f"{checkpoint_suffix()}",
    )
    os.makedirs(CKPT_PATH, exist_ok=True)

##########################################################################
############################# DATA PREPROCESSING #########################
##########################################################################
class DataOutput(NamedTuple):
    data_all: Any
    basal_mask: List[bool]
    idxgall: List[int]
    scale: Any

def load_data(data_path: str, load_extra:bool=False) -> DataOutput:
    global ACTIVE_REGIONS, FLOATING_REGIONS, KFAC_ACTIVE_REGIONS, KFAC_DATA_REGIONS, FREEZE_NETWORKS, KFAC_LOSS_TERMS
    print(f"[2] Loading and normalizing data from: {data_path}")
    rawdata = loadmat(data_path)

    basal_mask = [bool(b) for b in rawdata['basal_mask'].flatten()]
    print(f" . . . basal_mask (all regions): {basal_mask}")
    if USE_GROUNDED_ONLY_INTERFACE_MU_CT:
        assert len(basal_mask) == 2, 'grounded-only interface-mu-ct mode requires exactly two regions'
        ACTIVE_REGIONS = [idx for idx, is_basal in enumerate(basal_mask) if is_basal]
        FLOATING_REGIONS = [idx for idx, is_basal in enumerate(basal_mask) if not is_basal]
        assert len(ACTIVE_REGIONS) == 1 and len(FLOATING_REGIONS) == 1, 'grounded-only interface-mu-ct mode requires one grounded region and one floating region'
        KFAC_ACTIVE_REGIONS = list(ACTIVE_REGIONS)
        KFAC_DATA_REGIONS = list(ACTIVE_REGIONS)
        FREEZE_NETWORKS = freeze_regions_spec(FLOATING_REGIONS)
        KFAC_LOSS_TERMS = kfac_loss_terms()
    elif USE_FLOATING_REGION1_ONLY:
        assert len(basal_mask) > 1, 'floating-region-1-only mode requires at least two regions'
        assert not basal_mask[1], 'floating-region-1-only mode requires region 1 to be floating'
        grounded_regions = [idx for idx, is_basal in enumerate(basal_mask) if is_basal]
        assert len(grounded_regions) > 0, 'floating-region-1-only mode requires at least one grounded region to freeze'
        ACTIVE_REGIONS = [1]
        FLOATING_REGIONS = [1]
        KFAC_ACTIVE_REGIONS = [1]
        KFAC_DATA_REGIONS = [1]
        FREEZE_NETWORKS = freeze_regions_spec(grounded_regions)
        KFAC_LOSS_TERMS = kfac_loss_terms()
        print(f" . . . floating-region-1-only mode: active regions {ACTIVE_REGIONS}, frozen regions {grounded_regions}")
    elif USE_GROUNDED_REGION0_FROM_FLOATING:
        assert len(basal_mask) == 2, 'grounded-region-from-floating mode requires exactly two regions'
        grounded_regions = [idx for idx, is_basal in enumerate(basal_mask) if is_basal]
        floating_regions = [idx for idx, is_basal in enumerate(basal_mask) if not is_basal]
        assert len(grounded_regions) == 1 and len(floating_regions) == 1, 'grounded-region-from-floating mode requires one grounded and one floating region'
        ACTIVE_REGIONS = grounded_regions
        FLOATING_REGIONS = floating_regions
        KFAC_ACTIVE_REGIONS = list(ACTIVE_REGIONS)
        KFAC_DATA_REGIONS = list(ACTIVE_REGIONS)
        FREEZE_NETWORKS = freeze_regions_spec(FLOATING_REGIONS)
        KFAC_LOSS_TERMS = kfac_loss_terms()
        print(f" . . . grounded-from-floating mode: active regions {ACTIVE_REGIONS}, frozen regions {FLOATING_REGIONS}, KFAC terms {KFAC_LOSS_TERMS}")
    else:
        ACTIVE_REGIONS = list(range(len(basal_mask)))
        FLOATING_REGIONS = [idx for idx, is_basal in enumerate(basal_mask) if not is_basal]
        KFAC_ACTIVE_REGIONS = list(ACTIVE_REGIONS)
        KFAC_DATA_REGIONS = list(ACTIVE_REGIONS)
        FREEZE_NETWORKS = dict(net_u=[], net_mu=[], net_c0=[])
        KFAC_LOSS_TERMS = kfac_loss_terms()
    REGION_INDICES = list(range(len(basal_mask)))

    cell_keys = ['xd', 'yd', 'ud', 'vd',
                'xd_h', 'yd_h', 'hd', 'sd',
                'xct', 'yct', 'nnct',
                'xdir', 'ydir', 'udir', 'vdir',
                'xcol', 'ycol', 'h_dense', 's_dense']
    for key in cell_keys:
        if key in rawdata and rawdata[key].dtype == object:
            rawdata[key] = rawdata[key][:, REGION_INDICES]

    idxcrop_orig = rawdata['idxcrop']
    idxcrop_h_orig = rawdata['idxcrop_h']

    data_all, idxgall, posi_all, idxcrop_all = normdata_xpinn(
        rawdata,
        basal_mask=basal_mask,
        use_regression=True,
        grounded_only_interface_mu_ct=USE_GROUNDED_ONLY_INTERFACE_MU_CT,
    )
    scale:List[SubScaleResult] = tree_map(lambda x: data_all[x][4][6], idxgall)
    opt_data = {'h_dense': rawdata['h_dense'], 's_dense': rawdata['s_dense']}
    if load_extra:
        return DataOutput(data_all, basal_mask, idxgall, scale), opt_data
    else:
        return DataOutput(data_all, basal_mask, idxgall, scale)

##########################################################################
########################### XPINN INITIALIZATION #########################
##########################################################################
class XPINNOutput(NamedTuple):
    params: ArrayLike
    sol_NN: Tuple[Callable]
    eval_f: Callable

def initialize_xpinn(keys, data_output: DataOutput,
                     embedding_config=None, network_config=None) -> XPINNOutput:
    _, basal_mask, idxgall, scale, = data_output
    n_hl = 6
    n_unit = 30
    active_embedding_config = EMBEDDING_CONFIG if embedding_config is None else embedding_config
    active_network_config = NETWORK_CONFIG if network_config is None else network_config

    print(data_output.idxgall)
    print(active_embedding_config)
    print(active_network_config)
    subkey, keys = random.split(keys, 2)
    params = init_xpinn(
        subkey, n_hl, n_unit, n_sub=len(idxgall), basal_mask=basal_mask,
        embedding_config=active_embedding_config, network_config=active_network_config)
    pred_u, grad_u = solu_xpinn(scale, basal_mask=basal_mask)
    sol_NN = (pred_u, grad_u)
    print(len(scale))
    net = lambda params, x, idx: pred_u(params, x, idx)
    eqn_fn = lambda params, x, idx: ssa_iso(lambda z: net(params, z, idx), x, scale[idx], basal=basal_mask[idx])
    eval_f = lambda params, x, idx: jax.vmap(lambda z: eqn_fn(params, z, idx), in_axes=(0,))(x)

    return keys, XPINNOutput(params, sol_NN, eval_f)


##########################################################################
######################### LOSS FUNC INITIALIZATION #######################
##########################################################################
class LossOutput(NamedTuple):
    data_f: Callable
    loss_f: Callable
    initial_loss: float
    sol_f: Callable


class ResumeState(NamedTuple):
    step: int
    params: ArrayLike
    opt_state: Any
    optimizer: str
    damping: Any = None
    loss_history: Any = None
    freeze_spec: Any = None
    region_term_weights: Any = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=TRAIN_MODE_CHOICES, default=TRAIN_MODE)
    parser.add_argument('--optimizer', choices=('ADAM', 'KFAC'), default=OPTIMIZER)
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    parser.add_argument('--floating-checkpoint', type=str, default=None)
    return parser.parse_args()


def load_checkpoint(resume_checkpoint: str | None) -> ResumeState | None:
    if resume_checkpoint is None:
        return None

    ckpt_path = os.path.abspath(resume_checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if os.path.commonpath([REGRESSION_ROOT, ckpt_path]) != REGRESSION_ROOT:
        raise ValueError(f"Checkpoint must be inside {REGRESSION_ROOT}: {ckpt_path}")

    ckpt_name = os.path.basename(ckpt_path)
    if re.fullmatch(r'(KFAC_)?step_\d+\.pkl', ckpt_name) is None:
        raise ValueError(f"Resume checkpoint must match step_[num].pkl or KFAC_step_[num].pkl: {ckpt_name}")

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    required_keys = {'step', 'params', 'opt_state'}
    missing_keys = required_keys.difference(ckpt)
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {', '.join(sorted(missing_keys))}")

    step = int(ckpt['step'])
    optimizer_name = 'KFAC' if ckpt_name.startswith('KFAC_') else 'ADAM'
    print(f"[resume] Loaded {optimizer_name} checkpoint: {ckpt_path} (step {step})")
    return ResumeState(
        step=step,
        params=jax.device_put(ckpt['params']),
        opt_state=jax.device_put(ckpt['opt_state']),
        optimizer=optimizer_name,
        damping=jax.device_put(ckpt['damping']) if 'damping' in ckpt else None,
        loss_history=ckpt.get('loss_history'),
        freeze_spec=ckpt.get('freeze_spec'),
        region_term_weights=jax.device_put(ckpt['region_term_weights']) if 'region_term_weights' in ckpt else None,
    )


def load_checkpoint_params(checkpoint_path: str):
    ckpt_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if os.path.commonpath([REGRESSION_ROOT, ckpt_path]) != REGRESSION_ROOT:
        raise ValueError(f"Checkpoint must be inside {REGRESSION_ROOT}: {ckpt_path}")
    ckpt_name = os.path.basename(ckpt_path)
    if re.fullmatch(r'(KFAC_)?step_\d+\.pkl', ckpt_name) is None:
        raise ValueError(f"Floating checkpoint must match step_[num].pkl or KFAC_step_[num].pkl: {ckpt_name}")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    if 'params' not in ckpt:
        raise ValueError('Checkpoint missing required key: params')
    step = int(ckpt.get('step', -1))
    print(f"[floating] Loaded frozen floating checkpoint params: {ckpt_path} (step {step})")
    return jax.device_put(ckpt['params'])


def initialize_loss(keys, data_output: DataOutput, xpinn_output: XPINNOutput) -> LossOutput:
    data_all, basal_mask, idxgall, scale = data_output
    params, sol_NN, eval_f = xpinn_output

    lw = [1.0, 0.0, 0.0, 0.0]
    n_pt = REGRESSION_N_PT_BY_NSUB[len(idxgall)]

    data_f = dsample_regression_xpinn(
        data_all, idxgall, n_pt, basal_mask=basal_mask,
        grounded_only_interface_mu_ct=USE_GROUNDED_ONLY_INTERFACE_MU_CT)

    dkey, keys = random.split(keys, 2)
    smp = data_f(dkey)

    NN_loss = loss_regression_xpinn(sol_NN, idxgall, basal_mask=basal_mask,
                                    eqn=ssa_iso if USE_EQN else None,
                                    front_eqn=dbc_iso if EFFECTIVE_USE_CT else None,
                                    match=EFFECTIVE_USE_MATCHING,
                                    scales=data_output.scale,
                                    match_weight=matching_weight(0),
                                    match_component_weights=MATCH_COMPONENT_WEIGHTS_ON if USE_GROUNDED_REGION0_FROM_FLOATING else None,
                                    gpinn_weight=GPINN_WEIGHT if USE_GPINN else 0.0,
                                    mu_grad_weight=MU_GRAD_WEIGHT if USE_MU_GRAD_REG else 0.0,
                                    grounded_only_interface_mu_ct=USE_GROUNDED_ONLY_INTERFACE_MU_CT,
                                    active_regions=ACTIVE_REGIONS)

    smp = attach_loss_weights(smp, 0, idxgall)
    initial_loss = NN_loss(params, smp)[0]
    NN_loss.lref = initial_loss
    return keys, LossOutput(data_f, NN_loss, initial_loss, sol_NN)

def optimize(keys, data: DataOutput, xpinn: XPINNOutput, loss: LossOutput,
             resume_state: ResumeState | None = None):
    start_learning_rate = 1e-3
    stage_label = 'ADAM'
    u_label = 'u'
    v_label = 'v'

    optimizer = optax.adam(start_learning_rate)
    initial_params = xpinn.params if resume_state is None else resume_state.params
    freeze_mask, freeze_spec = make_freeze_mask(initial_params, FREEZE_NETWORKS)
    frozen_params = initial_params
    print(f'[ADAM] freeze spec: {freeze_spec}')
    reset_opt_state = resume_state is not None and freeze_spec_changed(resume_state.freeze_spec, freeze_spec)
    if resume_state is None:
        params = xpinn.params
        opt_state = optimizer.init(params)
        adam_start_step = 0
    else:
        params = resume_state.params
        opt_state = optimizer.init(params) if reset_opt_state else resume_state.opt_state
        adam_start_step = resume_state.step + 1
        if reset_opt_state:
            print('[resume] Freeze spec changed; reinitialized ADAM optimizer state.')
    params = project_frozen_params(params, frozen_params, freeze_mask)

    calc_loss = lambda params, x: loss.loss_f(train_view_params(params, frozen_params, freeze_mask), x)[0]

    loss_history = [] if resume_state is None or resume_state.loss_history is None else list(resume_state.loss_history)

    def save_checkpoint(step, params, opt_state):
        ckpt = {
            'step': step,
            'params': jax.device_get(params),
            'opt_state': jax.device_get(opt_state),
            'loss_history': jax.device_get(loss_history),
            'freeze_spec': freeze_spec,
        }
        with open(os.path.join(CKPT_PATH, f'step_{step}.pkl'), 'wb') as f:
            pickle.dump(ckpt, f)


    @jax.jit
    def train_step(params, opt_state, smp):
        l, grads = jax.value_and_grad(calc_loss)(params, smp)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params = project_frozen_params(new_params, frozen_params, freeze_mask)
        return new_params, new_opt_state, l

    def print_subregion_errs(reg_err_list):
        data_err = reg_err_list[0]
        eqn_err = reg_err_list[1]
        md_err = reg_err_list[2]
        ct_err = reg_err_list[3]
        n_region = len(data.idxgall)
        for idx in range(len(data_err)):
            status = 'active' if idx in ACTIVE_REGIONS else 'frozen'
            subregion_data_errs = data_err[idx]
            subregion_eqn_errs = eqn_err[idx]
            subregion_ct_errs = ct_err[idx]
            if idx==0:
                subregion_md_errs = md_err[idx]
            elif idx==n_region-1:
                subregion_md_errs = md_err[idx-1]
            else:
                subregion_md_errs = md_err[idx]

            print('')
            print(f'                  Region {idx} ({'grounded' if data.basal_mask[idx] else 'floating'}, {status}): {u_label} = {subregion_data_errs[0]:<5.3e} | {v_label} = {subregion_data_errs[1]:<5.3e} | h = {subregion_data_errs[2]:<5.3e} | s = {subregion_data_errs[3]:<5.3e} | mu= {subregion_data_errs[4]:<5.3e} | c = {subregion_data_errs[5]:<5.3e}')
            print(f'                            eqn: {format_eqn_errs(subregion_eqn_errs, data.basal_mask[idx])}')
            print(f'                            ct : {format_ct_errs(subregion_ct_errs, data.basal_mask[idx])}')
            for line in format_region_match_errs(idx, subregion_md_errs):
                print(f'                            {line}')

    x_col_mem = None
    adapted = False

    for step in range(adam_start_step, ADAM_MAXITER):
        dkey, keys = random.split(keys, 2)
        run_RAD = USE_ADAPTIVE_SAMPLING and (step + 1) % ADAPT_PERIOD == 0 and (step + 1) > ADAPT_BURNIN
        if run_RAD:
            smp = loss.data_f(
                dkey,
                eval_adaptive=True,
                eval_f=active_adaptive_eval(
                    lambda x, idx, basal: xpinn.eval_f(
                        train_view_params(params, frozen_params, freeze_mask), x, idx))
            )
            x_col_mem = smp['col'][0]
            adapted = True
        elif USE_ADAPTIVE_SAMPLING and adapted:
            smp = loss.data_f(dkey, eval_adaptive=False)
            smp['col'][0] = x_col_mem
        else:
            smp = loss.data_f(dkey)
        smp = attach_loss_weights(smp, step, data.idxgall)
        params, opt_state, l = train_step(params, opt_state, smp)
        loss_history.append(l)

        if step % 100 == 0:
            _, loss_info, reg_err_list = loss.loss_f(train_view_params(params, frozen_params, freeze_mask), smp)
            print(f'--------------------------------- STEP {step} ------------------------------------')
            print(f'{stage_label} step {step}: loss={loss_info[LOSS_INFO_TOTAL_IDX]:.2e} | {format_data_loss_summary(loss_info, u_label, v_label)}')
            print(f'                  {format_eqn_loss_summary(loss_info)}')
            print(f'                  {format_eqn_region_weights(smp)}')
            print(f'                  {format_gpinn_loss(loss_info)}')
            print(f'                  {format_mu_grad_loss(loss_info)}')
            print_subregion_errs(reg_err_list)
        if step % 1000 == 0:
            save_checkpoint(step, params, opt_state)

    return params, loss_history


def kfac_optimize(keys, kfac_config, data: DataOutput, xpinn: XPINNOutput, loss: LossOutput,
                  resume_state: ResumeState | None = None):
    if IS_METAL_BACKEND:
        raise ValueError("KFAC is unavailable on the JAX Metal backend. Run with --optimizer ADAM.")

    from kfac_jax import loss_functions as kfac_loss_functions
    from diffice_jax import KfacOptimizer

    data_output = data
    lossf = loss.loss_f
    dataf = loss.data_f
    stage_label = 'KFAC'
    u_label = 'u'
    v_label = 'v'
    initial_params = xpinn.params if resume_state is None else resume_state.params
    freeze_mask, freeze_spec = make_freeze_mask(initial_params, FREEZE_NETWORKS)
    frozen_params = initial_params
    print(f'[KFAC] freeze spec: {freeze_spec}')

    def kfac_eval_terms(params, data, terms, regions):
        return lossf.kfac_residuals(params, data, terms=terms, regions=regions)

    def kfac_residual_vector(params, data):
        residuals = [
            jnp.sqrt(KFAC_PHYS_OBJECTIVE_WEIGHT) * kfac_eval_terms(
                params, data,
                terms=KFAC_LOSS_TERMS,
                regions=KFAC_ACTIVE_REGIONS,
            )
        ]
        if USE_DATA_IN_KFAC:
            residuals.append(
                jnp.sqrt(KFAC_DATA_OBJECTIVE_WEIGHT) * kfac_eval_terms(
                    params, data,
                    terms=('data',),
                    regions=KFAC_DATA_REGIONS,
                )
            )
        return concat_residuals(residuals)

    def residual_objective(residuals):
        return jnp.sum(jnp.square(residuals)) / lossf.lref

    def kfac_objective_value(params, data):
        return residual_objective(kfac_residual_vector(params, data))

    def kfac_lossf(params, data):
        active_params = train_view_params(params, frozen_params, freeze_mask)
        weighted_residuals = kfac_residual_vector(active_params, data)
        loss_n = residual_objective(weighted_residuals)
        residuals = weighted_residuals / jnp.sqrt(lossf.lref)
        kfac_loss_functions.register_squared_error_loss(
            residuals,
            targets=jnp.zeros_like(residuals),
        )
        return loss_n, jnp.array([loss_n, loss_n, loss_n])

    optim = KfacOptimizer(
        loss_fn=kfac_lossf, **kfac_config).get_optimizer()

    rng, init_key = jax.random.split(keys)
    kfac_start_step = 0 if resume_state is None else resume_state.step + 1
    init_data = attach_loss_weights(dataf(init_key), kfac_start_step, data_output.idxgall)
    region_term_weights = (
        initialize_region_term_weights(data_output.idxgall, data_output.basal_mask)
        if resume_state is None or resume_state.region_term_weights is None
        else resume_state.region_term_weights
    )
    if USE_REGION_TERM_BALANCING:
        init_data['region_term_weights'] = region_term_weights
    reset_opt_state = resume_state is not None and freeze_spec_changed(resume_state.freeze_spec, freeze_spec)
    if resume_state is None:
        params = xpinn.params
        opt_state = optim.init(params, init_key, init_data)
    else:
        params = resume_state.params
        opt_state = optim.init(params, init_key, init_data) if reset_opt_state else resume_state.opt_state
        if reset_opt_state:
            print('[resume] Freeze spec changed; reinitialized KFAC optimizer state.')
    params = project_frozen_params(params, frozen_params, freeze_mask)
    loss_all = [] if resume_state is None or resume_state.loss_history is None else list(resume_state.loss_history)
    losse_all = []
    damping = kfac_config['initial_damping'] if resume_state is None or resume_state.damping is None else resume_state.damping
    damping_decay = kfac_config['damping_adaptation_decay']
    damping_min = kfac_config['min_damping']
    x_col_mem = None
    adapted = False

    def save_checkpoint(step, params, opt_state, damping):
        ckpt = {
            'step': step,
            'params': jax.device_get(params),
            'opt_state': jax.device_get(opt_state),
            'damping': jax.device_get(damping),
            'loss_history': jax.device_get(loss_all),
            'freeze_spec': freeze_spec,
            'region_term_weights': jax.device_get(region_term_weights),
        }
        with open(os.path.join(CKPT_PATH, f'KFAC_step_{step}.pkl'), 'wb') as f:
            pickle.dump(ckpt, f)

    def print_subregion_errs(reg_err_list):
        data_err = reg_err_list[0]
        eqn_err = reg_err_list[1]
        md_err = reg_err_list[2]
        ct_err = reg_err_list[3]
        n_region = len(data_output.idxgall)
        for idx in range(len(data_err)):
            status = 'active' if idx in ACTIVE_REGIONS else 'frozen'
            subregion_data_errs = data_err[idx]
            subregion_eqn_errs = eqn_err[idx]
            subregion_ct_errs = ct_err[idx]
            if idx == 0:
                subregion_md_errs = md_err[idx]
            elif idx == n_region - 1:
                subregion_md_errs = md_err[idx - 1]
            else:
                subregion_md_errs = md_err[idx]

            print('')
            print(f'                  Region {idx} ({'grounded' if data_output.basal_mask[idx] else 'floating'}, {status}): {u_label} = {subregion_data_errs[0]:<5.3e} | {v_label} = {subregion_data_errs[1]:<5.3e} | h = {subregion_data_errs[2]:<5.3e} | s = {subregion_data_errs[3]:<5.3e} | mu= {subregion_data_errs[4]:<5.3e} | c = {subregion_data_errs[5]:<5.3e}')
            print(f'                            eqn: {format_eqn_errs(subregion_eqn_errs, data_output.basal_mask[idx])}')
            print(f'                            ct : {format_ct_errs(subregion_ct_errs, data_output.basal_mask[idx])}')
            for line in format_region_match_errs(idx, subregion_md_errs):
                print(f'                            {line}')

    # start the training iteration
    for step in range(kfac_start_step, KFAC_MAXITER):
        rng, step_key, data_key = jax.random.split(rng, 3)
        run_RAD = USE_ADAPTIVE_SAMPLING and (step + 1) % ADAPT_PERIOD == 0 and (step + 1) > ADAPT_BURNIN
        if run_RAD:
            print(f'[KFAC] step {step + 1}: adapting collocation sample based on equation residual')
            data = dataf(
                data_key,
                eval_adaptive=True,
                eval_f=active_adaptive_eval(
                    lambda x, idx, basal: xpinn.eval_f(
                        train_view_params(params, frozen_params, freeze_mask), x, idx))
            )
            x_col_mem = data['col'][0]
            adapted = True
        elif USE_ADAPTIVE_SAMPLING and adapted:
            data = dataf(data_key, eval_adaptive=False)
            data['col'][0] = x_col_mem
        else:
            data = dataf(data_key)
        data = attach_loss_weights(data, step, data_output.idxgall)
        if USE_REGION_TERM_BALANCING:
            if step % REGION_TERM_UPDATE_PERIOD == 0:
                region_term_weights = compute_region_term_weights(
                    params, data, lossf, data_output.idxgall, data_output.basal_mask,
                    frozen_params, freeze_mask, region_term_weights,
                )
            data['region_term_weights'] = region_term_weights
        params, opt_state, stats = optim.step(
            params, opt_state, step_key, batch=data, damping=damping, global_step_int=step)
        params = project_frozen_params(params, frozen_params, freeze_mask)

        loss_info = stats['aux']
        loss_all.append(loss_info)
        losse_all.append(loss_info[2])

        if step % KFAC_LOG_RATE == 0:
            _, loss_info, reg_err_list = loss.loss_f(train_view_params(params, frozen_params, freeze_mask), data)
            loss_all[-1] = loss_info
            active_params = train_view_params(params, frozen_params, freeze_mask)
            dmp = stats['damping']
            mdw = float(data['match_weight'])
            kfac_phys = residual_objective(
                jnp.sqrt(KFAC_PHYS_OBJECTIVE_WEIGHT) * kfac_eval_terms(
                    active_params, data,
                    terms=KFAC_LOSS_TERMS,
                    regions=KFAC_ACTIVE_REGIONS,
                )
            )
            kfac_data = residual_objective(
                jnp.sqrt(KFAC_DATA_OBJECTIVE_WEIGHT) * kfac_eval_terms(
                    active_params, data,
                    terms=('data',),
                    regions=KFAC_DATA_REGIONS,
                )
            ) if USE_DATA_IN_KFAC else 0.0
            print(f'--------------------------------- STEP {step} ------------------------------------')
            print(f'{stage_label} step {step}: loss={loss_info[LOSS_INFO_TOTAL_IDX]:.2e} | {format_data_loss_summary(loss_info, u_label, v_label)} | Dp={dmp:.2e} | md_w={mdw:.2e}')
            print(f'                  kfac_phys={kfac_phys:.2e} (w={KFAC_PHYS_OBJECTIVE_WEIGHT:.1e}) | kfac_data={kfac_data:.2e} (w={KFAC_DATA_OBJECTIVE_WEIGHT:.1e})')
            print(f'                  {format_eqn_loss_summary(loss_info)}')
            print(f'                  {format_eqn_region_weights(data)}')
            print(f'                  {format_region_term_weights(data, data_output)}')
            print(f'                  {format_gpinn_loss(loss_info)}')
            print(f'                  {format_mu_grad_loss(loss_info)}')
            print_subregion_errs(reg_err_list)

        if step % KFAC_CKPT_RATE == 0:
            save_checkpoint(step, params, opt_state, damping)

        if damping > damping_min:
            damping *= damping_decay

    return params, loss_all

if __name__ == '__main__':
    wall_start = time.time()
    args = parse_args()
    set_train_mode(args.mode)
    refresh_checkpoint_path()
    print(f'[mode] TRAIN_MODE={TRAIN_MODE} | EFFECTIVE_USE_MATCHING={EFFECTIVE_USE_MATCHING} | EFFECTIVE_USE_CT={EFFECTIVE_USE_CT} | KFAC_LOSS_TERMS={KFAC_LOSS_TERMS}')
    print(f'[mode] Checkpoints will be saved to: {CKPT_PATH}')
    keys = random.PRNGKey(8132002)

    data_output = load_data(DATA_PATH)
    keys, xpinn_output = initialize_xpinn(keys, data_output)
    resume_state = load_checkpoint(args.resume_checkpoint)
    if resume_state is not None and resume_state.optimizer != args.optimizer:
        raise ValueError(f"Resume checkpoint is for {resume_state.optimizer}, but --optimizer={args.optimizer}")
    if USE_GROUNDED_REGION0_FROM_FLOATING and resume_state is None:
        if args.floating_checkpoint is None:
            raise ValueError('--floating-checkpoint is required for grounded_region0_from_floating mode')
        floating_params = load_checkpoint_params(args.floating_checkpoint)
        xpinn_output = XPINNOutput(floating_params, xpinn_output.sol_NN, xpinn_output.eval_f)
    elif USE_GROUNDED_REGION0_FROM_FLOATING and args.floating_checkpoint is not None:
        print('[floating] Resume checkpoint supplied; using resume params and ignoring --floating-checkpoint for initialization.')
    if resume_state is not None:
        xpinn_output = XPINNOutput(resume_state.params, xpinn_output.sol_NN, xpinn_output.eval_f)
    keys, loss_output = initialize_loss(keys, data_output, xpinn_output)
    print(f'Initial X-PINN Loss = {loss_output.initial_loss:.2e}')
    if args.optimizer == 'ADAM':
        params, loss_history = optimize(
            keys, data_output, xpinn_output, loss_output,
            resume_state=resume_state)
    else:
        params, loss_history = kfac_optimize(
            keys, kfac_config, data_output, xpinn_output, loss_output,
            resume_state=resume_state)
    with open(os.path.join(CKPT_PATH, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(loss_history, f)
    wall_elapsed = time.time() - wall_start
    print(f'Total wall time: {wall_elapsed:.2f} s ({wall_elapsed / 60:.2f} min)')
