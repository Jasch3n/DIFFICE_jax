"""
@author: Yongji Wang
"""

import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import jax.lax as lax
import jax.debug as jdb
from typing import NamedTuple, Tuple
from jax.typing import ArrayLike

N_INTERFACE_LIBRARY = 1500
N_INTERFACE_COLLOCATION = 600

class DataSample(NamedTuple):
    X_smp: ArrayLike 
    U_smp: ArrayLike
    Xh_smp: ArrayLike 
    H_smp: ArrayLike
    S_smp: ArrayLike 
    Mu_smp: ArrayLike 
    C_smp: ArrayLike
    
def eval_RAD_probs(X_col_all, idxgall, basal_mask, eval_f, return_diagnostics=False):
    # RAD pdf with k=2 and c=1 (see Wu et al. 2023)
    def compute_pdf(err_item):
        err_sq = jnp.sum(jnp.square(err_item), axis=1)
        err_sq = jnp.where(jnp.isfinite(err_sq), err_sq, 0.0)
        err_mean = jnp.mean(err_sq)
        p = jnp.where(err_mean > 0.0, err_sq / err_mean + 1.0, jnp.ones_like(err_sq))
        p = jnp.where(jnp.isfinite(p), p, 1.0)
        p_sum = jnp.sum(p)
        return jnp.where(p_sum > 0.0, p / p_sum, jnp.ones_like(p) / p.shape[0])

    def compute_diagnostic(err_item, probs_item, region_idx):
        err_sq = jnp.sum(jnp.square(err_item), axis=1)
        err_sq = jnp.where(jnp.isfinite(err_sq), err_sq, 0.0)
        err_norm = jnp.sqrt(err_sq)
        idx_min_res = jnp.argmin(err_norm)
        idx_max_res = jnp.argmax(err_norm)
        idx_min_prob = jnp.argmin(probs_item)
        idx_max_prob = jnp.argmax(probs_item)
        res_min = err_norm[idx_min_res]
        res_max = err_norm[idx_max_res]
        prob_min = probs_item[idx_min_prob]
        prob_max = probs_item[idx_max_prob]
        eps = jnp.asarray(jnp.finfo(err_norm.dtype).eps, dtype=err_norm.dtype)
        res_roundoff = eps * jnp.maximum(res_max, eps)
        prob_roundoff = eps * jnp.maximum(prob_max, eps)
        return {
            "region": int(region_idx),
            "eps": eps,
            "res_min": res_min,
            "res_max": res_max,
            "prob_at_res_min": probs_item[idx_min_res],
            "prob_at_res_max": probs_item[idx_max_res],
            "res_min_roundoff_ratio": res_min / res_roundoff,
            "prob_min": prob_min,
            "prob_max": prob_max,
            "prob_span_roundoff_ratio": (prob_max - prob_min) / prob_roundoff,
        }

    def normalized_eqn_err(x_col, idx, basal):
        eqn_out = eval_f(x_col, idx, basal)
        err = eqn_out[0]
        if len(eqn_out) < 2:
            return err
        if basal:
            x_term_ref = jnp.hstack((eqn_out[1][:, 0:3], eqn_out[1][:, 7:8]))
            y_term_ref = jnp.hstack((eqn_out[1][:, 3:6], eqn_out[1][:, 8:9]))
        else:
            x_term_ref = eqn_out[1][:, 0:3]
            y_term_ref = eqn_out[1][:, 3:6]
        x_term_scale = jnp.max(jnp.abs(x_term_ref), axis=1, keepdims=True)
        y_term_scale = jnp.max(jnp.abs(y_term_ref), axis=1, keepdims=True)
        x_term_scale = jnp.where(jnp.isfinite(x_term_scale) & (x_term_scale > 0.0), x_term_scale, 1.0)
        y_term_scale = jnp.where(jnp.isfinite(y_term_scale) & (y_term_scale > 0.0), y_term_scale, 1.0)
        return err / jnp.hstack((x_term_scale, y_term_scale))

    eqn_err = [
        normalized_eqn_err(X_col_all[pos], idx, basal_mask[pos])
        for pos, idx in enumerate(idxgall)
    ]
    probs = [compute_pdf(err_item) for err_item in eqn_err]
    if return_diagnostics:
        diagnostics = [
            compute_diagnostic(err_item, probs_item, idx)
            for err_item, probs_item, idx in zip(eqn_err, probs, idxgall)
        ]
        return probs, diagnostics
    return probs

def nearest_interface_collocation_library(X_col, X_if):
    n_take = min(N_INTERFACE_LIBRARY, X_col.shape[0])
    if n_take == 0 or X_if.shape[0] == 0:
        return jnp.zeros((0, X_col.shape[1]), dtype=X_col.dtype)
    dist_sq = jnp.min(jnp.sum(jnp.square(X_col[:, None, :] - X_if[None, :, :]), axis=2), axis=1)
    idx = jnp.argsort(dist_sq)[:n_take]
    return X_col[idx]

def interface_collocation_libraries(X_col_all, X_md, basal_mask):
    X_interface_lib = [
        jnp.zeros((0, X_col_all[pos].shape[1]), dtype=X_col_all[pos].dtype)
        for pos in range(len(X_col_all))
    ]
    for pos in range(len(X_md)):
        if basal_mask[pos] == basal_mask[pos + 1]:
            continue
        X_interface_lib[pos] = jnp.vstack([
            X_interface_lib[pos],
            nearest_interface_collocation_library(X_col_all[pos], X_md[pos][:, 0:2]),
        ])
        X_interface_lib[pos + 1] = jnp.vstack([
            X_interface_lib[pos + 1],
            nearest_interface_collocation_library(X_col_all[pos + 1], X_md[pos][:, 2:4]),
        ])
    return X_interface_lib

def append_interface_collocation(X_col, X_interface):
    return [
        jnp.vstack([X_col[pos], X_interface[pos]])
        if X_interface[pos].shape[0] > 0 else X_col[pos]
        for pos in range(len(X_col))
    ]

def sample_interface_collocation(keys, X_interface_lib):
    X_interface = []
    for pos, X_lib in enumerate(X_interface_lib):
        if X_lib.shape[0] == 0:
            X_interface.append(X_lib)
            continue
        idx = random.choice(
            keys[pos],
            jnp.arange(X_lib.shape[0]),
            [N_INTERFACE_COLLOCATION],
            replace=X_lib.shape[0] < N_INTERFACE_COLLOCATION,
        )
        X_interface.append(X_lib[idx])
    return X_interface

def data_sample_create(data_all, idxgall:ArrayLike, n_pt: ArrayLike, basal_mask=None, use_regression=False):
    # obtain the number of sub-group
    ng = len(idxgall)
    if basal_mask is None:
        basal_mask = [False] * ng
    elif len(basal_mask) != ng:
        raise ValueError('[xpinn :: sampling] basal_mask must match the number of sub-regions')
    # load the data within each sub-region
    X_star = tree_map(lambda x: data_all[x][0], idxgall)
    U_star = tree_map(lambda x: data_all[x][1], idxgall)
    # load the data at the calving fronts (for ice shelf regions only)
    X_ct = tree_map(lambda x: data_all[x][2], idxgall)
    nn_ct = tree_map(lambda x: data_all[x][3], idxgall)
    Xraw_md = tree_map(lambda x: data_all[x][5], idxgall)
    X_md = Xraw_md[0:-1]
    n_md = [jnp.array(1.)] * (ng-1)

    # For adaptive sampling later
    X_col_all = tree_map(lambda x: X_star[x][2], idxgall)
    
    n_pt_norm = []
    
    expected_lens = [ng, ng, ng, ng, max(ng-1, 1), ng] 
    # n_pt input might be array or list.
    # We will convert to valid list of lists.
    
    for k in range(len(n_pt)):
        val = n_pt[k]
        exp_len = expected_lens[k] if k < len(expected_lens) else ng
        
        # Check if scalar
        if hasattr(val, 'ndim') and val.ndim == 0:
            # JAX scalar or numpy scalar
            val_list = [int(val)] * exp_len
        elif isinstance(val, (int, float)):
             val_list = [int(val)] * exp_len
        elif isinstance(val, (list, tuple)) or (hasattr(val, 'ndim') and val.ndim > 0):
            # It's iterable
             if len(val) == 1:
                 val_list = [int(val[0])] * exp_len
             else:
                 val_list = [int(v) for v in val]
                 # Ideally check len(val_list) == exp_len
        else:
             val_list = [int(val)] * exp_len
        
        # Special handling for interface (index 4) if ng=1 (empty list expected or dummy?)
        # code uses idxgall[0:-1]. If ng=1, this is empty.
        # tree_map will loop over empty lists. So n_pt[4] length doesn't matter much if empty.
        # But if ng > 1, we need it to match.
        
        n_pt_norm.append(val_list)
    
    n_pt = n_pt_norm # Override n_pt with normalized version

    # load the data at the sub-region boundary
    # X_md is always at index 5 in the normalize_each output tuple
    # (index -1 may be boundary_star for basal regions)
    # load the data at the connect
    for l in range(ng - 1):
        # obtain the boundary from the previous subregion
        if l == 0:
            X_md1 = Xraw_md[l]
        else:
            n_md0 = n_md[l - 1]
            X_md1 = Xraw_md[l][n_md0:]
        # obtain the boundary from the next subregion
        n_md1 = X_md1.shape[0]
        X_md2 = Xraw_md[l + 1][0:n_md1, :]
        # pair the boundary in both sub-regions
        X_mdp = jnp.hstack([X_md1, X_md2])
            
        n_md[l] = n_md1
        X_md[l] = X_mdp
    X_interface_lib = interface_collocation_libraries(X_col_all, X_md, basal_mask)

    # create the index of velocity data points within all sub-regions
    idx_data = tree_map(lambda x: jnp.arange(X_star[x][0].shape[0]), idxgall)
    # create the index of thickness data points within all sub-regions
    idxh_data = tree_map(lambda x: jnp.arange(X_star[x][1].shape[0]), idxgall)
    # create the index of collocation points within all sub-regions
    idx_col_data = tree_map(lambda x: jnp.arange(X_star[x][2].shape[0]), idxgall)
    # create the index of data points for all sub-regions at the calving front
    # For grounded regions with no calving front, X_ct has shape (0, 2), yielding an empty index
    idx_bd = tree_map(lambda x: jnp.arange(max(X_ct[x].shape[0], 1)), idxgall)
    # create the index of data points at the interface between different pairs of sub-regions
    idx_md = tree_map(lambda x: jnp.arange(X_md[x].shape[0]), idxgall[0:-1])

    # define the function that can re-sampling for each calling
    def dataf(key, eval_adaptive=None, eval_f=None):
        # generate the new random key
        _, *keys = random.split(key, 5*ng + 1)

        # sampling the velocity data point based on the index
        idx_smp = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=False), keys[0:ng], idx_data, n_pt[0])
        X_smp = tree_map(lambda x, y: X_star[x][0][y], idxgall, idx_smp)
        U_smp = tree_map(lambda x, y: U_star[x][0][y], idxgall, idx_smp)

        # sampling the thickness data point based on the index
        idxh_smp = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=False), keys[0:ng], idxh_data, n_pt[1])
        Xh_smp = tree_map(lambda x, y: X_star[x][1][y], idxgall, idxh_smp)
        H_smp = tree_map(lambda x, y: U_star[x][1][y], idxgall, idxh_smp)

        # sampling the surface elevation data for basal regions (same indices as thickness)
        # U_star[x] has 3 elements [uv, h, s] for basal, 2 [uv, h] for floating
        S_smp = tree_map(lambda x, y: U_star[x][2][y], idxgall, idxh_smp)
        
        if use_regression:
            Mu_smp = tree_map(lambda x, y: U_star[x][3][y], idxgall, idx_smp)
            C_smp  = tree_map(lambda x, y: U_star[x][4][y], idxgall, idx_smp)

        # Sample collocation points, potentially based on equation residuals
        rad_diagnostics = None
        if eval_adaptive:
            probs, rad_diagnostics = eval_RAD_probs(
                X_col_all, idxgall, basal_mask, eval_f, return_diagnostics=True)
        else:
            probs = None
        idx_col = [
            random.choice(
                keys[ng + pos],
                idx_col_data[pos],
                [n_pt[2][pos]],
                p=None if probs is None else probs[pos],
                replace=True
            )
            for pos in range(ng)
        ]
        X_col = tree_map(lambda x, y: X_star[x][2][y], idxgall, idx_col)
        X_interface_col = sample_interface_collocation(keys[(4*ng):(5*ng)], X_interface_lib)
        X_col = append_interface_collocation(X_col, X_interface_col)

        # Generate a random index of the data at ice front
        idx_cbd = tree_map(lambda x, y, n: random.choice(x, y, [n]), keys[(2*ng):(3*ng)], idx_bd, n_pt[3])
        # For regions with empty boundary data (grounded), create zero-filled placeholders
        def safe_bd_sample(xct, idx, n):
            if xct.shape[0] == 0:
                return jnp.zeros((n, xct.shape[1]))
            return xct[idx]
        X_bd = tree_map(lambda x, y, n: safe_bd_sample(X_ct[x], y, n), idxgall, idx_cbd, n_pt[3])
        
        def safe_nn_sample(nnct, idx, n):
            if nnct.shape[0] == 0:
                return jnp.zeros((n, nnct.shape[1]))
            return nnct[idx]
        nn_bd = tree_map(lambda x, y, n: safe_nn_sample(nn_ct[x], y, n), idxgall, idx_cbd, n_pt[3])

        # generate a random index of the data at matching boundary
        idx_mbd = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=False), keys[(3*ng):(4*ng-1)], idx_md, n_pt[4])
        # sampling the data point based on the index
        X_mbd = tree_map(lambda x, y: X_md[x][y], idxgall[0:-1], idx_mbd)

        # group all the data and collocation points
        if use_regression:
            sample = DataSample(X_smp, U_smp, Xh_smp, H_smp, S_smp, Mu_smp, C_smp)
        else:
            sample = DataSample(X_smp, U_smp, Xh_smp, H_smp, S_smp, [], [])
            
        data = dict(smp=sample, col=[X_col],  bd=[X_bd, nn_bd], md=[X_mbd])
        if rad_diagnostics is not None:
            data["rad_diagnostics"] = rad_diagnostics
        
        return data
    
    return dataf


def data_regression_sample_create(data_all, idxgall:ArrayLike, n_pt: ArrayLike, basal_mask=None,
                                  grounded_only_interface_mu_ct:bool=False):
    # obtain the number of sub-group
    ng = len(idxgall)
    if basal_mask is None:
        basal_mask = [False] * ng
    elif len(basal_mask) != ng:
        raise ValueError('[xpinn :: regression sampling] basal_mask must match the number of sub-regions')
    # load the data within each sub-region
    X_star = tree_map(lambda x: data_all[x][0], idxgall)
    U_star = tree_map(lambda x: data_all[x][1], idxgall)
    Xraw_md = tree_map(lambda x: data_all[x][5], idxgall)
    X_md = Xraw_md[0:-1]
    n_md = [jnp.array(1.)] * (ng-1)
    X_ct_star = tree_map(lambda x: data_all[x][2], idxgall)
    nnct_star = tree_map(lambda x: data_all[x][3], idxgall)
    bd_star = tree_map(lambda x: data_all[x][6] if len(data_all[x]) > 6 else None, idxgall)
    X_col_all = tree_map(lambda x: X_star[x][2], idxgall)

    for l in range(ng - 1):
        if l == 0:
            X_md1 = Xraw_md[l]
        else:
            n_md0 = n_md[l - 1]
            X_md1 = Xraw_md[l][n_md0:]
        n_md1 = X_md1.shape[0]
        X_md2 = Xraw_md[l + 1][0:n_md1, :]
        X_md[l] = jnp.hstack([X_md1, X_md2])
        n_md[l] = n_md1
    X_interface_lib = interface_collocation_libraries(X_col_all, X_md, basal_mask)

    # create the index of velocity data points within all sub-regions
    idx_data = tree_map(lambda x: jnp.arange(X_star[x][0].shape[0]), idxgall)
    idxh_data = tree_map(lambda x: jnp.arange(X_star[x][1].shape[0]), idxgall)
    idx_col_data = tree_map(lambda x: jnp.arange(X_star[x][2].shape[0]), idxgall)
    n_col = n_pt[2] if len(n_pt) > 2 else n_pt[0]

    # define the function that can re-sampling for each calling
    def dataf(key, eval_adaptive=None, eval_f=None):
        # generate the new random key
        _, *keys = random.split(key, 5*ng + 1)

        # sampling the velocity data point based on the index
        idx_smp = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=False), keys[0:ng], idx_data, n_pt[0])
        X_smp = tree_map(lambda x, y: X_star[x][0][y], idxgall, idx_smp)
        U_smp = tree_map(lambda x, y: U_star[x][0][y], idxgall, idx_smp)

        # sampling the thickness data point based on the index
        idxh_smp = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=False), keys[0:ng], idxh_data, n_pt[1])
        Xh_smp = tree_map(lambda x, y: X_star[x][1][y], idxgall, idxh_smp)
        H_smp = tree_map(lambda x, y: U_star[x][1][y], idxgall, idxh_smp)

        # sampling the surface elevation data for basal regions (same indices as thickness)
        # U_star[x] has 3 elements [uv, h, s] for basal, 2 [uv, h] for floating
        S_smp = tree_map(lambda x, y: U_star[x][2][y], idxgall, idxh_smp)
        
        Mu_smp = tree_map(lambda x, y: U_star[x][3][y], idxgall, idx_smp)
        C_smp  = tree_map(lambda x, y: U_star[x][4][y], idxgall, idx_smp)
        # jdb.print('[DEBUG]: Region 0 Avg C_smp = {s}', s=jnp.mean(C_smp[0]))
        # jdb.print('[DEBUG]: Region 1 Avg C_smp = {s}', s=jnp.mean(C_smp[1]))
        # jdb.print('[DEBUG]: Region 2 Avg C_smp = {s}', s=jnp.mean(C_smp[2]))
        
        sample = DataSample(X_smp, U_smp, Xh_smp, H_smp, S_smp, Mu_smp, C_smp)
        
        rad_diagnostics = None
        if eval_adaptive:
            probs, rad_diagnostics = eval_RAD_probs(
                X_col_all, idxgall, basal_mask, eval_f, return_diagnostics=True)
        else:
            probs = None
        idx_col = [
            random.choice(
                keys[ng + pos],
                idx_col_data[pos],
                [n_col[pos]],
                p=None if probs is None else probs[pos],
                replace=True
            )
            for pos in range(ng)
        ]
        X_col_smp = tree_map(lambda x, y: X_star[x][2][y], idxgall, idx_col)
        X_interface_col = sample_interface_collocation(keys[(4*ng):(5*ng)], X_interface_lib)
        X_col_smp = append_interface_collocation(X_col_smp, X_interface_col)

        X_ct = tree_map(lambda x: X_ct_star[x], idxgall)
        if grounded_only_interface_mu_ct:
            nnct = tree_map(
                lambda x: jnp.zeros((X_ct_star[x].shape[0], 1))
                if bd_star[x] is None else bd_star[x][0],
                idxgall)
        else:
            nnct = tree_map(lambda x: nnct_star[x], idxgall)
        X_md_smp = tree_map(lambda x: X_md[x], idxgall[:-1])
        # jdb.print('X_md len = {s}', s=len(X_md_smp))
        # jdb.print('X_md shape = {s}', s=X_md_smp[0].shape)
            
        data = dict(smp=sample, ct=[X_ct, nnct], col=[X_col_smp], md=[X_md_smp])
        if rad_diagnostics is not None:
            data["rad_diagnostics"] = rad_diagnostics
        
        return data
    
    return dataf
