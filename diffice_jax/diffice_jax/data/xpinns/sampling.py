"""
@author: Yongji Wang
"""

import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import jax.lax as lax
import jax.debug as jdb

def eval_RAD_probs(X_col_all, eval_f):
    f_pred_1 = eval_f(X_col_all[0], 0, False)[1]
    f_pred_2 = eval_f(X_col_all[1], 1, True)[1]
    eqn_err = [f_pred_1, f_pred_2]

    # RAD pdf with k=2 and c=1 (see Wu et al. 2023)
    def compute_pdf(err_item):
        err_sq = jnp.sum(jnp.square(err_item[:, 0:6]), axis=1)
        p = err_sq / jnp.mean(err_sq) + 1
        p /= jnp.sum(p)
        return p

    probs = tree_map(lambda x: compute_pdf(eqn_err[x]), [0, 1])
    return probs

def data_sample_create(data_all, idxgall, n_pt):

    def sample_s(x, idx):
            if len(U_star[x]) > 2:
                return U_star[x][2][idx]
            return None
    
    # obtain the number of sub-group
    ng = len(idxgall)
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
    X_col_all = tree_map(lambda x: X_star[x][0], idxgall)
    
    # Normalize n_pt to be list of lists (or arrays) matching subregions
    # n_pt structure: [vel, thick, col, boundary, interface, (adapt)]
    # Ensure each is a list of length ng (or ng-1 for interface)
    n_pt_norm = []
    # indices 0 (vel), 1 (thick), 2 (col), 3 (bd), 4 (interface), 5 (adapt)
    # mapping to expected lengths: 
    # 0,1,2,3,5 -> ng
    # 4 -> ng-1 (if ng > 1)
    
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

    # create the index of velocity data points within all sub-regions
    idx_data = tree_map(lambda x: jnp.arange(X_star[x][0].shape[0]), idxgall)
    # create the index of thickness data points within all sub-regions
    idxh_data = tree_map(lambda x: jnp.arange(X_star[x][1].shape[0]), idxgall)
    # create the index of data points for all sub-regions at the calving front
    # For grounded regions with no calving front, X_ct has shape (0, 2), yielding an empty index
    idx_bd = tree_map(lambda x: jnp.arange(max(X_ct[x].shape[0], 1)), idxgall)
    # create the index of data points at the interface between different pairs of sub-regions
    idx_md = tree_map(lambda x: jnp.arange(X_md[x].shape[0]), idxgall[0:-1])

    # define the function that can re-sampling for each calling
    def dataf(key, eval_adaptive=None, eval_f=None):
        # generate the new random key
        _, *keys = random.split(key, 4*ng)

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
        S_smp = tree_map(lambda x, y: sample_s(x, y), idxgall, idxh_smp)

        # generate a random sample of collocation point within the domain
        # Check if eval_adaptive is enabled (globally or per subregion)
        # If eval_adaptive=True (global), we return full grids for all regions (for RAD eval).
        # If eval_adaptive is a list/dict, check per region.
        
        # Helper to decide sampling strategy for a region
        probs = None
        if eval_adaptive: 
            probs = eval_RAD_probs(X_col_all, eval_f)
            
        def sample_col(key, region_idx, region_indices, n_c, n_a):        
            if (not probs is None):
                return random.choice(key, region_indices, [n_c], p=probs[region_idx], replace=True)
            else:
                # Uniform sampling
                return random.choice(key, region_indices, [n_c], replace=False)

        # Apply sampling per region
        # Pass n_pt[2] (collocation counts) and n_pt[5] (adapt counts if exists)
        n_a_list = n_pt[5] if len(n_pt) > 5 else [0]*ng
        idx_col = tree_map(lambda k, i, indices, n_c, n_a: sample_col(k, i, indices, n_c, n_a), 
                           keys[ng:(2*ng)], 
                           idxgall, 
                           idx_data,
                           n_pt[2],
                           n_a_list)
        X_col = tree_map(lambda x, y: X_star[x][0][y], idxgall, idx_col)

        # Generate a random index of the data at ice front
        # [NOTE]: Use replace=True to handle grounded regions with fewer boundary points than requested
        idx_cbd = tree_map(lambda x, y, n: random.choice(x, y, [n], replace=True), keys[(2*ng):(3*ng)], idx_bd, n_pt[3])
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
        data = dict(smp=[X_smp, U_smp, Xh_smp, H_smp, S_smp], col=[X_col],  bd=[X_bd, nn_bd], md=[X_mbd])
        return data
    return dataf


