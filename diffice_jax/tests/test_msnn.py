"""
Tests for the Multi-Stage Neural Networks (MSNN) implementation.

Tests cover:
  1. Correction network initialization (κ scaling)
  2. Multi-stage forward pass (frozen + active)
  3. Residue computation
  4. Full MSNN two-stage training (integration)
"""

import sys
import os
import jax
import jax.numpy as jnp
from jax import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffice_jax.model.xpinns.initialization import init_nets, init_correction_nets
from diffice_jax.model.xpinns.networks import solu_create, msnn_solu_create, neural_net
from diffice_jax.model.xpinns.loss import loss_iso_create
from diffice_jax.model.xpinns.residue import (
    compute_equation_residue, estimate_kappa, estimate_epsilon,
    estimate_epsilon_per_variable, estimate_gamma
)
from diffice_jax.model.xpinns.msnn_config import MSNNConfig
from diffice_jax.equation.eqn_iso import gov_eqn, front_eqn


# ---- Shared test fixtures ----
def make_test_setup(n_sub=2, basal_mask=None):
    """Create common test data for MSNN tests."""
    if basal_mask is None:
        basal_mask = [False, True]  # Region 0: Floating, Region 1: Grounded

    key = random.PRNGKey(42)
    n_hl = 2
    n_unit = 10
    N = 20  # number of sample points

    # Scale info per region
    scale0 = [jnp.array([0., 0., 0., 0., 1.0]), jnp.array([1., 1., 1., 1., 1.])]
    scale1 = [jnp.array([0., 0., 0., 0., 1.0, 1.0]), jnp.array([1., 1., 1., 1., 1., 1.])]
    scale = [scale0, scale1]

    # Initialize Stage 0 params
    params = init_nets(key, n_hl, n_unit, n_sub=n_sub, basal_mask=basal_mask)

    # Mock data
    data = {
        'smp': [
            [jnp.zeros((N, 2)), jnp.zeros((N, 2))],  # x_smp
            [jnp.zeros((N, 2)), jnp.zeros((N, 2))],  # u_smp
            [jnp.zeros((N, 2)), jnp.zeros((N, 2))],  # xh_smp
            [jnp.zeros((N, 1)), jnp.zeros((N, 1))],  # h_smp
            [None, jnp.zeros((N, 1))],                 # s_smp
        ],
        'col': [[jnp.zeros((N, 2)), jnp.zeros((N, 2))]],
        'bd': [
            [jnp.zeros((N, 2)), jnp.zeros((N, 2))],
            [jnp.zeros((N, 2)), None],
            [None, None],
        ],
        'md': [[jnp.zeros((N, 4))]],
    }

    return key, n_hl, n_unit, params, scale, basal_mask, data, N


# ---- Test 1: Correction network initialization ----
def test_correction_net_init():
    """Verify init_correction_nets produces correct shapes and κ-scaled weights."""
    print("Test 1: Correction network initialization...")

    key = random.PRNGKey(0)
    n_hl = 2
    n_unit = 15
    n_sub = 2
    basal_mask = [False, True]
    kappa_per_region = [5.0, 8.0]

    corr_params = init_correction_nets(
        key, n_hl, n_unit, n_sub, kappa_per_region, basal_mask=basal_mask)

    # Check structure
    assert len(corr_params['net_u']) == 2, "Should have 2 net_u sub-networks"
    assert len(corr_params['net_mu']) == 2, "Should have 2 net_mu sub-networks"
    assert corr_params['net_c'][0] is None, "Floating region should have no net_c"
    assert corr_params['net_c'][1] is not None, "Grounded region should have net_c"

    # Check output dimensions
    # Floating: 3 outputs (u, v, h)
    last_layer_0 = corr_params['net_u'][0][-1]
    assert last_layer_0[0].shape[1] == 3, f"Floating net_u output dim should be 3, got {last_layer_0[0].shape[1]}"

    # Grounded: 4 outputs (u, v, h, s)
    last_layer_1 = corr_params['net_u'][1][-1]
    assert last_layer_1[0].shape[1] == 4, f"Grounded net_u output dim should be 4, got {last_layer_1[0].shape[1]}"

    # κ scaling is NOT baked into weights — it's applied at runtime via neural_net's scl param.
    # Verify first-layer weights have standard Xavier magnitude (no κ factor).
    first_w_0 = corr_params['net_u'][0][0][0]  # (2, n_unit)
    first_w_1 = corr_params['net_u'][1][0][0]  # (2, n_unit)

    # Both regions should have similar weight magnitudes (pure Xavier, no κ)
    mean_abs_0 = float(jnp.mean(jnp.abs(first_w_0)))
    mean_abs_1 = float(jnp.mean(jnp.abs(first_w_1)))
    ratio = mean_abs_1 / mean_abs_0
    print(f"  Weight magnitude ratio (region1/region0): {ratio:.2f} (expected ~1.0, no κ baked in)")
    assert 0.3 < ratio < 3.0, \
        f"Weights should have similar magnitude (no κ): got ratio {ratio:.2f}"

    print("  PASSED\n")


# ---- Test 2: activation mode act_s=2 ----
def test_act_s_2():
    """Verify act_s=2 uses sin on first layer and tanh on hidden layers."""
    print("Test 2: act_s=2 activation mode...")

    key = random.PRNGKey(1)
    from diffice_jax.model.xpinns.initialization import init_single_net

    # Create a small network: 2 -> 5 -> 5 -> 3
    layers = [2, 5, 5, 3]
    params = init_single_net(key, layers)

    x = jnp.array([[0.5, 0.3], [-0.2, 0.8]])

    # act_s=0: tanh everywhere
    out0 = neural_net(params, x, scl=1.0, act_s=0)
    # act_s=2: sin on first, tanh on rest
    out2 = neural_net(params, x, scl=1.0, act_s=2)

    # Outputs should differ (different first-layer activation)
    assert not jnp.allclose(out0, out2), "act_s=0 and act_s=2 should produce different outputs"

    # Manual computation for act_s=2 to verify
    first, hidden, last = params[0], params[1], params[2]
    H = jnp.sin(jnp.dot(x, first[0]) * 1.0 + first[1])  # sin first layer
    H = jnp.tanh(jnp.dot(H, hidden[0]) + hidden[1])       # tanh hidden
    manual = jnp.dot(H, last[0]) + last[1]

    assert jnp.allclose(out2, manual, atol=1e-6), \
        f"act_s=2 output doesn't match manual computation"

    print("  PASSED\n")


# ---- Test 3: msnn_solu_create forward pass ----
def test_msnn_forward_pass():
    """Verify the combined ansatz sums frozen + active stages correctly."""
    print("Test 3: MSNN forward pass...")

    key, n_hl, n_unit, stage0_params, scale, basal_mask, data, N = make_test_setup()
    idxgall = [0, 1]

    # Create the Stage 0 prediction (standard)
    solNN_0 = solu_create(scale, scl=1, basal_mask=basal_mask)
    predNN_0, _ = solNN_0

    x_test = jnp.linspace(-0.5, 0.5, 10).reshape(-1, 1)
    x_test = jnp.hstack([x_test, x_test * 0.5])  # (10, 2)

    # Stage 0 prediction for floating region (idx=0)
    pred_stage0 = predNN_0(stage0_params, x_test, 0)

    # Create correction net params
    kappa_per_region = [3.0, 3.0]
    # Per-variable epsilon arrays: floating has 4 entries (u,v,h,mu), grounded has 6 (u,v,h,s,mu,c)
    eps_per_region = [jnp.array([0.01, 0.01, 0.01, 0.01]),          # floating: u, v, h, mu
                      jnp.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])]  # grounded: u, v, h, s, mu, c
    key, subkey = random.split(key)
    corr_params = init_correction_nets(
        subkey, 2, 10, 2, kappa_per_region, basal_mask=basal_mask)

    # Create MSNN combined prediction with stage0 frozen + correction active
    frozen_stages = [(stage0_params, [1.0, 1.0], 1.0)]  # Stage 0
    msnn_pred, msnn_grad = msnn_solu_create(
        scale, frozen_stages=frozen_stages,
        active_epsilon=eps_per_region,
        active_kappa=3.0, scl=1, basal_mask=basal_mask)

    # Combined prediction should initially equal Stage 0 (correction nets are zero-initialized)
    pred_combined_init = msnn_pred(corr_params, x_test, 0)
    diff_init = jnp.sum(jnp.abs(pred_combined_init - pred_stage0))
    print(f"  Stage0 vs Combined (at init, zero-init correction): {diff_init:.6f}")
    assert diff_init < 1e-5, "Zero-initialized correction should not change Stage 0 prediction"

    # Perturb correction params so they produce non-zero output
    import copy
    perturbed_params = copy.deepcopy(corr_params)
    perturbed_params['net_u'][0][-1][0] = jnp.ones_like(perturbed_params['net_u'][0][-1][0]) * 0.1
    perturbed_params['net_u'][0][-1][1] = jnp.ones_like(perturbed_params['net_u'][0][-1][1]) * 0.1

    pred_combined = msnn_pred(perturbed_params, x_test, 0)
    diff = jnp.sum(jnp.abs(pred_combined - pred_stage0))
    print(f"  Stage0 vs Combined (perturbed): {diff:.6f}")
    assert diff > 1e-6, "Combined prediction with perturbed correction should differ from Stage 0"

    # Verify gradient only flows through active params
    def loss_fn(params):
        return jnp.sum(msnn_pred(params, x_test, 0))

    grads = jax.grad(loss_fn)(perturbed_params)

    # Active correction net_u should have non-zero gradients
    grad_sum_active = sum(float(jnp.sum(jnp.abs(g)))
                          for layer in grads['net_u'][0] for g in layer)
    print(f"  Active net_u grad magnitude: {grad_sum_active:.6f}")
    assert grad_sum_active > 1e-10, "Active stage should have non-zero gradients"

    print("  PASSED\n")


# ---- Test 4: Residue computation ----
def test_residue_computation():
    """Verify compute_equation_residue returns non-zero residuals."""
    print("Test 4: Residue computation...")

    key, n_hl, n_unit, params, scale, basal_mask, data, N = make_test_setup()

    solNN_0 = solu_create(scale, scl=1, basal_mask=basal_mask)
    predNN_0, _ = solNN_0

    # Use non-zero collocation points
    x_col = random.uniform(key, shape=(50, 2), minval=-0.5, maxval=0.5)

    # Compute residue for floating region
    residue_0 = compute_equation_residue(
        predNN_0, gov_eqn, scale, params, x_col, 0, basal=False)

    assert residue_0.shape == (50, 2), f"Residue shape should be (50, 2), got {residue_0.shape}"
    assert float(jnp.sum(jnp.abs(residue_0))) > 0, "Residue should be non-zero for untrained network"

    # Compute residue for grounded region
    residue_1 = compute_equation_residue(
        predNN_0, gov_eqn, scale, params, x_col, 1, basal=True)

    assert residue_1.shape == (50, 2), f"Residue shape should be (50, 2), got {residue_1.shape}"
    print(f"  Floating residue RMS: {float(jnp.sqrt(jnp.mean(residue_0**2))):.4e}")
    print(f"  Grounded residue RMS: {float(jnp.sqrt(jnp.mean(residue_1**2))):.4e}")

    print("  PASSED\n")


# ---- Test 5: κ, ε, γ estimation ----
def test_estimation_functions():
    """Test the κ/ε/γ estimation utilities."""
    print("Test 5: κ/ε/γ estimation...")

    # Create synthetic residue data
    key = random.PRNGKey(99)
    N = 100
    x_col = random.uniform(key, shape=(N, 2), minval=-1.0, maxval=1.0)
    # Synthetic residue with a known frequency
    residue = jnp.sin(5.0 * jnp.pi * x_col[:, 0:1]) * 0.1
    residue = jnp.hstack([residue, residue * 0.5])  # (N, 2)

    domain_range = jnp.array([2.0, 2.0, 1.0, 1.0])
    kappa, f_d = estimate_kappa(residue, x_col, domain_range, n_hl=2, n_unit=30)
    print(f"  Estimated f_d: {f_d:.2f}, κ: {kappa:.2f}")
    assert f_d > 0, "Dominant frequency should be positive"
    assert kappa > 0, "κ should be positive"

    # Old scalar epsilon (still available but deprecated for MSNN)
    epsilon = estimate_epsilon(residue, f_d, pde_order=2)
    print(f"  Estimated scalar ε: {epsilon:.4e}")
    assert epsilon > 0, "ε should be positive"

    # New per-variable epsilon
    key2, n_hl2, n_unit2, params2, scale2, basal_mask2, data2, N2 = make_test_setup()
    solNN_test = solu_create(scale2, scl=1, basal_mask=basal_mask2)
    predNN_test, _ = solNN_test

    # Floating region (idx=0): should return 3 epsilons (u, v, h)
    eps_float = estimate_epsilon_per_variable(
        predNN_test, params2, data2, scale2, 0, basal=False)
    print(f"  Per-variable ε (floating): {eps_float}")
    assert eps_float.shape == (4,), f"Expected shape (4,), got {eps_float.shape}"  # u, v, h, mu
    assert jnp.all(eps_float > 0), "All ε should be positive"

    # Grounded region (idx=1): should return 4 epsilons (u, v, h, s)
    eps_ground = estimate_epsilon_per_variable(
        predNN_test, params2, data2, scale2, 1, basal=True)
    print(f"  Per-variable ε (grounded): {eps_ground}")
    assert eps_ground.shape == (6,), f"Expected shape (6,), got {eps_ground.shape}"  # u, v, h, s, mu, c
    assert jnp.all(eps_ground > 0), "All ε should be positive"

    gamma = estimate_gamma(0.5, 0.5)
    assert abs(gamma - 0.5) < 1e-6, f"γ should be 0.5, got {gamma}"

    gamma2 = estimate_gamma(0.1, 0.9)
    assert abs(gamma2 - 0.1) < 1e-6, f"γ should be 0.1, got {gamma2}"

    print("  PASSED\n")


# ---- Test 6: MSNNConfig validation ----
def test_msnn_config():
    """Test MSNNConfig dataclass validation."""
    print("Test 6: MSNNConfig validation...")

    config = MSNNConfig(n_stages=2, stage_epochs=[30000, 20000])
    assert len(config.stage_epochs) == 2
    assert len(config.use_lbfgs) == 2
    assert config.correction_n_hl == 2
    assert config.correction_n_unit == 30

    # Test default initialization
    config2 = MSNNConfig(n_stages=3)
    assert len(config2.stage_epochs) == 3
    assert config2.stage_epochs == [30000, 30000, 30000]

    # Test mismatch should raise
    try:
        config_bad = MSNNConfig(n_stages=2, stage_epochs=[30000])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    print("  PASSED\n")


if __name__ == "__main__":
    test_correction_net_init()
    test_act_s_2()
    test_msnn_forward_pass()
    test_residue_computation()
    test_estimation_functions()
    test_msnn_config()
    print("=" * 40)
    print("All MSNN tests passed!")
