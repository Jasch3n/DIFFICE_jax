import jax.numpy as jnp
from jax import random
import jax
from sklearn.neighbors import NearestNeighbors
import jax.debug as jdb

def eval_RAD_probs(X_col_lib, eval_f, basal=False):
    f_pred = eval_f(X_col_lib, 0, basal)[1]
    
    # RAD pdf with k=2 and c=1 (see Wu et al. 2023)
    def compute_pdf(err_item):
        err_sq = jnp.sum(jnp.square(err_item[:, 0:6]), axis=1)
        p = err_sq / jnp.mean(err_sq) + 1
        p /= jnp.sum(p)
        return p

    probs = compute_pdf(f_pred)
    return probs

# wrapper to create function that can re-sample the dataset and collocation points
def data_sample_create(data_all, n_pt,basal=False):
    # load the data within ice shelf
    X_star = data_all[0]
    U_star = data_all[1]
    # if basal:
    #     ocean_mask = data_all[5]
    # load the data at the ice front
    X_ct = data_all[2]
    nn_ct = data_all[3]
    if basal:
        u_bd, v_bd, h_bd, mu_bd = data_all[5]
    # obtain the number of data points and points at the boundary
    n_data = X_star[0].shape[0]
    nh_data = X_star[1].shape[0]
    n_bd = X_ct.shape[0]

    # ================== EXPAND COLLOCATION POINT LIBRARY ==================
    def expand_col_lib(X_col_library, K, M=4*n_data):
        print(f' . . . . . . Expanding collocation point library from {n_data} to {M} points')
        nn = NearestNeighbors(n_neighbors=K).fit(X_col_library)
        distances, indices = nn.kneighbors(X_col_library)
        key_aug = random.PRNGKey(0)  # base key; will be overridden per-call inside dataf if needed

        # For each point, randomly pick one of its K neighbors and interpolate
        key_aug, k1, k2 = random.split(random.PRNGKey(42), 3)

        # Draw random neighbor indices (which neighbor to interpolate toward) for all library points
        neighbor_choices = random.randint(k1, shape=(M,), minval=1, maxval=K)  # skip self (index 0)
        base_choices = random.choice(k2, jnp.arange(n_data), shape=(M,))

        # Random interpolation weights in [0, 1]
        key_aug, k3 = random.split(key_aug)
        alphas = random.uniform(k3, shape=(M, 1))

        # Gather base points and their chosen neighbors
        X_lib = jnp.array(X_col_library)
        base_pts = X_lib[base_choices]                                           # (M, d)
        neighbor_idx = jnp.array(indices)[base_choices, neighbor_choices]        # (M,)
        neighbor_pts = X_lib[neighbor_idx]                                       # (M, d)

        # Interpolate: new_point = alpha * base + (1 - alpha) * neighbor
        X_col_lib = alphas * base_pts + (1 - alphas) * neighbor_pts       # (M, d)
        return X_col_lib
    
    M = 6000
    X_col_lib = expand_col_lib(X_star[0], 40, M=M)

    # define the function that can re-sampling for each calling
    def dataf(key, eval_adaptive=False, adaptive_probs=None, adapt_data=False, mix_adaptive=False, eval_f=None):
        # generate the new random key
        keys = random.split(key, 4)

        if eval_adaptive and eval_f is not None:
            adaptive_probs = eval_RAD_probs(X_col_lib, eval_f, basal=basal)

        # generate a random sample of collocation point within the domain
        if adaptive_probs is None:
            idx_col = random.choice(keys[2], jnp.arange(M), [n_pt[2]], replace=False)
        elif (not adaptive_probs is None) and mix_adaptive:
            idx_col_1 = random.choice(keys[2], jnp.arange(M), [n_pt[2]], replace=False)
            # n_pt[4] would be out of bounds if n_pt has len 4, safe fallback to n_pt[2]
            n_adapt = n_pt[4] if len(n_pt) > 4 else n_pt[2]
            idx_col_2 = random.choice(keys[2], jnp.arange(M), [n_adapt], p=adaptive_probs, replace=True)
            idx_col = jnp.concatenate((idx_col_1, idx_col_2), axis=0)
        else:
            idx_col = random.choice(keys[2], jnp.arange(M), [n_pt[2]], p=adaptive_probs, replace=True)
        X_col = X_col_lib[idx_col]

        # sampling the velocity data point based on the index
        if adapt_data:
            X_smp = X_star[0][idx_col]
            U_smp = U_star[0][idx_col]
        else:
            idx_smp = random.choice(keys[0], jnp.arange(n_data), [n_pt[0]])
            X_smp = X_star[0][idx_smp]
            U_smp = U_star[0][idx_smp]

        # sampling the thickness data point based on the index
        idxh_smp = random.choice(keys[1], jnp.arange(nh_data), [n_pt[1]])
        Xh_smp = X_star[1][idxh_smp]
        H_smp = U_star[1][idxh_smp]
        if basal:
            S_smp = U_star[2][idxh_smp]

        # sampling the boundary based on index
        idx_cbd = random.choice(keys[3], jnp.arange(n_bd), [n_pt[3]])
        X_bd = X_ct[idx_cbd]
        if basal: 
            mu_bd_smp = mu_bd[idx_cbd]
            h_bd_smp = h_bd[idx_cbd]
            u_bd_smp = jnp.expand_dims(u_bd[idx_cbd], axis=1)
            v_bd_smp = jnp.expand_dims(v_bd[idx_cbd], axis=1)
        else:
            nn_bd = nn_ct[idx_cbd]

        # group all the data and collocation points
        if basal:
            data = dict(smp=[X_smp, U_smp, Xh_smp, H_smp, S_smp], col=[X_col],  bd=[X_bd, jnp.hstack((u_bd_smp, v_bd_smp)), h_bd_smp, mu_bd_smp])
        else: 
            data = dict(smp=[X_smp, U_smp, Xh_smp, H_smp], col=[X_col],  bd=[X_bd, nn_bd])
        return data
    return dataf

