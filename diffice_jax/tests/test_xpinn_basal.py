
import sys
import os
import jax
import jax.numpy as jnp
from jax import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffice_jax.model.xpinns.initialization import init_nets
from diffice_jax.model.xpinns.networks import solu_create
from diffice_jax.model.xpinns.loss import loss_iso_create
from diffice_jax.equation.eqn_iso import gov_eqn, front_eqn

def test_xpinn_basal():
    print("Testing XPINN Basal Inversion...")
    
    # Setup
    key = random.PRNGKey(0)
    n_hl = 2
    n_unit = 10
    n_sub = 2
    basal_mask = [False, True] # Region 0: Floating, Region 1: Basal
    
    # Initialize Networks
    print("Initializing networks...")
    params = init_nets(key, n_hl, n_unit, n_sub=n_sub, basal_mask=basal_mask)
    
    # Check params structure
    assert len(params['net_u']) == 2
    assert len(params['net_mu']) == 2
    assert len(params['net_c']) == 2
    assert params['net_c'][0] is None
    assert params['net_c'][1] is not None
    print("Params structure verified.")

    # Mock Scale Info
    # dmean: [x_mean, y_mean, u_mean, v_mean, h_mean, (s_mean)]
    # drange: [x_range, y_range, u_range, v_range, h_range, (s_range)]
    scale0 = [jnp.array([0., 0., 0., 0., 1.0]), jnp.array([1., 1., 1., 1., 1.])]
    scale1 = [jnp.array([0., 0., 0., 0., 1.0, 1.0]), jnp.array([1., 1., 1., 1., 1., 1.])]
    scale = [scale0, scale1]

    # Create Solution Function
    solNN = solu_create(scale, basal_mask=basal_mask)
    
    # Mock Data
    N = 10
    # Floating region data
    x_smp0 = jnp.zeros((N, 2))
    u_smp0 = jnp.zeros((N, 2))
    xh_smp0 = jnp.zeros((N, 2))
    h_smp0 = jnp.zeros((N, 1))
    
    # Basal region data
    x_smp1 = jnp.zeros((N, 2))
    u_smp1 = jnp.zeros((N, 2))
    xh_smp1 = jnp.zeros((N, 2))
    h_smp1 = jnp.zeros((N, 1))
    s_smp1 = jnp.zeros((N, 1))
    
    # Collocation points
    x_col0 = jnp.zeros((N, 2))
    x_col1 = jnp.zeros((N, 2))
    
    # Boundary data
    x_bd0 = jnp.zeros((N, 2))
    nn_bd0 = jnp.zeros((N, 2)) # Floating boundary normal
    
    x_bd1 = jnp.zeros((N, 2))
    
    # Matching data
    x_md0 = jnp.zeros((N, 4)) # [x1 y1 x2 y2]
    
    data = {
        'smp': [
            [x_smp0, x_smp1],
            [u_smp0, u_smp1],
            [xh_smp0, xh_smp1],
            [h_smp0, h_smp1],
            [None, s_smp1] # s data. Region 0 has None? In loss.py: s_smp = data['smp'][4][idx]. IF idx=0 (floating), it is NOT accessed. So None is fine.
        ],
        'col': [
            [x_col0, x_col1]
        ],
        'bd': [
            [x_bd0, x_bd1],
            [nn_bd0, None], # index 1 is nn_bd for floating. Ignored for Basal.
            [None, None]   # index 2 is mu_bd for basal. Removed.
        ],
        'md': [
            [x_md0]
        ]
    }
    
    # Create Loss Function
    eqn_all = (gov_eqn, front_eqn)
    idxgall = [0, 1]
    lw = [1.0, 1.0, 1.0, 1.0] # Weights
    
    print("Creating loss function...")
    loss_fun = loss_iso_create(solNN, eqn_all, scale, idxgall, lw, basal_mask=basal_mask)
    
    # Compute Loss
    print("Computing loss...")
    loss, loss_info = loss_fun(params, data)
    
    print(f"Loss computed: {loss}")
    # loss_info is [info_array, residuals_list]
    info_array = loss_info[0]
    print(f"Loss info shape: {info_array.shape}")
    
    # Check matching loss (loss_md is the 5th element in info_array: [loss, loss_data, loss_eqn, loss_bd, loss_md, ...])
    loss_md_initial = info_array[4]
    print(f"Initial matching loss: {loss_md_initial}")
    
    # Create Data Function for Optimizer (optimizer expects dataf)
    def dataf(key, adaptive_probs=None, eval_adaptive=False):
        if eval_adaptive:
            # return full data for adaptive evaluation
            return data
        # For standard training, return data (mocking sampling)
        return data

    # Run Optimizer
    print("\n--- Running Basal Two-Stage Optimizer ---")
    # from diffice_jax.optimizer.optimization import basal_twoStage_adam_optimizer
    from diffice_jax.optimizer.optimization import basal_twoStage_optimizer
    
    # Define stage epochs
    stage1_epochs = 10
    stage2_epochs = 10
    
    # Run optimizer
    # Note: we use small epochs to test mechanism
    # Updated signature: 
    # params, loss_all, params_stage1, (loss1_adam, loss1_lbfgs), params_stage2, (loss2_adam, loss2_lbfgs)
    
    params_optimized, loss_all, params_s1, (loss1, loss1_lbfgs), params_s2, (loss2, loss2_lbfgs) = basal_twoStage_optimizer(
        key, loss_fun, params, dataf,
        stage1_epochs, stage2_epochs,
        lw, lw, # Use same weights for simplicity in test
        basal_mask, lr=1e-2, # High LR to ensure visible updates
        adaptive=False,
        use_lbfgs=True, lbfgs_epochs=5, dataf_l=dataf # Use same dataf for L-BFGS test
    )
    
    print("\n--- Verifying Stage 1 (Floating Only) ---")
    # Region 0 is Floating. Region 1 is Grounded.
    # In Stage 1, Region 0 should update. Region 1 should be FROZEN.
    
    # Check Net U
    # Region 0
    # Use tree_map to get absolute difference, then sum leaves
    diff_tree_u0 = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), params_s1['net_u'][0], params['net_u'][0])
    diff_u0 = jax.tree_util.tree_reduce(lambda x, y: x + y, diff_tree_u0)
    print(f"Region 0 (Floating) Net U Change: {diff_u0}")
    
    # Region 1
    diff_tree_u1 = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), params_s1['net_u'][1], params['net_u'][1])
    diff_u1 = jax.tree_util.tree_reduce(lambda x, y: x + y, diff_tree_u1)
    print(f"Region 1 (Grounded) Net U Change: {diff_u1}")
    
    if diff_u0 > 1e-6:
        print("SUCCESS: Floating region updated in Stage 1.")
    else:
        print("FAILURE: Floating region DID NOT update in Stage 1.")
        
    if diff_u1 < 1e-9:
        print("SUCCESS: Grounded region validly frozen in Stage 1.")
    else:
        print(f"FAILURE: Grounded region UPDATED in Stage 1 (Diff: {diff_u1}).")

    print("\n--- Verifying Stage 2 (Grounded Only) ---")
    # In Stage 2, Region 0 should be FROZEN (relative to end of Stage 1). Region 1 should UPDATE.
    
    # Check Net U
    # Region 0 (Compare params_s2 vs params_s1)
    diff_tree_u0_s2 = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), params_s2['net_u'][0], params_s1['net_u'][0])
    diff_u0_s2 = jax.tree_util.tree_reduce(lambda x, y: x + y, diff_tree_u0_s2)
    print(f"Region 0 (Floating) Net U Change in S2: {diff_u0_s2}")
    
    # Region 1 (Compare params_s2 vs params_s1)
    diff_tree_u1_s2 = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), params_s2['net_u'][1], params_s1['net_u'][1])
    diff_u1_s2 = jax.tree_util.tree_reduce(lambda x, y: x + y, diff_tree_u1_s2)
    print(f"Region 1 (Grounded) Net U Change in S2: {diff_u1_s2}")
    
    if diff_u0_s2 < 1e-9:
        print("SUCCESS: Floating region validly frozen in Stage 2.")
    else:
        print(f"FAILURE: Floating region UPDATED in Stage 2 (Diff: {diff_u0_s2}).")
        
    if diff_u1_s2 > 1e-6:
        print("SUCCESS: Grounded region updated in Stage 2.")
    else:
        print("FAILURE: Grounded region DID NOT update in Stage 2.")

    # Check Loss History
    print("\n--- Checking Loss History ---")
    loss1_vals = [l[0] for l in loss1]
    print(f"Stage 1 Adam Loss values: {loss1_vals}")
    
    print(f"Stage 1 L-BFGS Loss values: {loss1_lbfgs}")
    if len(loss1_lbfgs) > 0:
        print("SUCCESS: Stage 1 L-BFGS ran.")
    else:
        print("FAILURE: Stage 1 L-BFGS did not run.")

    loss2_vals = [l[0] for l in loss2]
    print(f"Stage 2 Adam Loss values: {loss2_vals}")

    print(f"Stage 2 L-BFGS Loss values: {loss2_lbfgs}")
    if len(loss2_lbfgs) > 0:
        print("SUCCESS: Stage 2 L-BFGS ran.")
    else:
        print("FAILURE: Stage 2 L-BFGS did not run.")

    print("Test completed.")

if __name__ == "__main__":
    test_xpinn_basal()
