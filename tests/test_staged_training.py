"""Tests for two-stage (basal) training: gradient masking and loss weight switching."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from jax import random

from diffice_jax.model.xpinns.initialization import init_nets
from diffice_jax.model.xpinns.networks import solu_create
from diffice_jax.model.xpinns.loss import loss_iso_create
from diffice_jax.equation.eqn_iso import gov_eqn, front_eqn
from diffice_jax.optimizer.optimization import build_grad_mask, basal_twoStage_adam_optimizer


def _make_mock_setup():
    """Create a minimal 2-subregion XPINN (floating + grounded) with mock data."""
    key = random.PRNGKey(42)
    n_hl, n_unit, n_sub = 2, 10, 2
    basal_mask = [False, True]

    params = init_nets(key, n_hl, n_unit, n_sub=n_sub, basal_mask=basal_mask)

    scale0 = [jnp.array([0., 0., 0., 0., 1.0]),
              jnp.array([1., 1., 1., 1., 1.])]
    scale1 = [jnp.array([0., 0., 0., 0., 1.0, 1.0]),
              jnp.array([1., 1., 1., 1., 1., 1.])]
    scale = [scale0, scale1]

    N = 10
    data = {
        'smp': [
            [jnp.ones((N, 2)) * 0.1, jnp.ones((N, 2)) * 0.2],
            [jnp.ones((N, 2)) * 0.5, jnp.ones((N, 2)) * 0.3],
            [jnp.ones((N, 2)) * 0.1, jnp.ones((N, 2)) * 0.2],
            [jnp.ones((N, 1)) * 0.8, jnp.ones((N, 1)) * 0.6],
            [None, jnp.ones((N, 1)) * 0.9],
        ],
        'col': [[jnp.ones((N, 2)) * 0.1, jnp.ones((N, 2)) * 0.2]],
        'bd': [
            [jnp.ones((N, 2)) * 0.1, jnp.ones((N, 2)) * 0.2],
            [jnp.ones((N, 2)) * 0.5, None],
            [None, None],
        ],
        'md': [[jnp.ones((N, 4)) * 0.15]],
    }
    return params, scale, data, basal_mask, n_sub


def test_build_grad_mask():
    """Test that build_grad_mask correctly freezes the right regions."""
    print("--- test_build_grad_mask ---")
    params, scale, data, basal_mask, n_sub = _make_mock_setup()

    # Freeze grounded (idx=1)
    mask_fg = build_grad_mask(params, basal_mask, freeze_grounded=True)
    # Floating (idx=0) should be 1.0
    for layer in mask_fg['net_u'][0]:
        for arr in layer:
            assert jnp.all(arr == 1.0), "Floating net_u should be trainable"
    # Grounded (idx=1) should be 0.0
    for layer in mask_fg['net_u'][1]:
        for arr in layer:
            assert jnp.all(arr == 0.0), "Grounded net_u should be frozen"
    # net_c[0] is None (floating has no friction net)
    assert mask_fg['net_c'][0] is None
    # net_c[1] should be frozen (grounded frozen)
    for layer in mask_fg['net_c'][1]:
        for arr in layer:
            assert jnp.all(arr == 0.0), "Grounded net_c should be frozen"

    # Freeze floating (idx=0)
    mask_ff = build_grad_mask(params, basal_mask, freeze_grounded=False)
    for layer in mask_ff['net_u'][0]:
        for arr in layer:
            assert jnp.all(arr == 0.0), "Floating net_u should be frozen"
    for layer in mask_ff['net_u'][1]:
        for arr in layer:
            assert jnp.all(arr == 1.0), "Grounded net_u should be trainable"

    print("  PASSED: gradient masks are correct")


def test_gradient_masking():
    """Verify that masked minimizer actually freezes the correct params."""
    print("--- test_gradient_masking ---")
    params, scale, data, basal_mask, n_sub = _make_mock_setup()
    idxgall = list(range(n_sub))
    lw = [1.0, 1.0, 1.0, 1.0]
    eqn_all = (gov_eqn, front_eqn)
    solNN = solu_create(scale, basal_mask=basal_mask)

    loss_fun = loss_iso_create(solNN, eqn_all, scale, idxgall, lw, basal_mask=basal_mask)
    # Set lref
    loss_fun.lref = loss_fun(params, data)[0]

    # Build mask: freeze grounded
    mask = build_grad_mask(params, basal_mask, freeze_grounded=True)

    # Save original grounded params
    orig_net_u_1 = jax.tree_util.tree_map(lambda x: x.copy(), params['net_u'][1])
    orig_net_mu_1 = jax.tree_util.tree_map(lambda x: x.copy(), params['net_mu'][1])

    # One gradient step with masking
    import optax, functools
    from jax import jit, grad

    @functools.partial(jit, static_argnames=("lossf", "opt"))
    def masked_step(lossf, params, data, opt, opt_state, grad_mask):
        grads, loss_info = grad(lossf, has_aux=True)(params, data)
        grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss_info, opt_state

    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    new_params, _, _ = masked_step(loss_fun, params, data, opt, opt_state, mask)

    # Check grounded params unchanged
    for orig_layer, new_layer in zip(orig_net_u_1, new_params['net_u'][1]):
        for orig_arr, new_arr in zip(orig_layer, new_layer):
            assert jnp.allclose(orig_arr, new_arr), "Grounded net_u should not change"

    for orig_layer, new_layer in zip(orig_net_mu_1, new_params['net_mu'][1]):
        for orig_arr, new_arr in zip(orig_layer, new_layer):
            assert jnp.allclose(orig_arr, new_arr), "Grounded net_mu should not change"

    # Check floating params DID change
    floating_changed = False
    for orig_layer, new_layer in zip(params['net_u'][0], new_params['net_u'][0]):
        for orig_arr, new_arr in zip(orig_layer, new_layer):
            if not jnp.allclose(orig_arr, new_arr):
                floating_changed = True
    assert floating_changed, "Floating net_u should have changed"

    print("  PASSED: gradient masking correctly freezes/unfreezes params")


def test_loss_weight_mutation():
    """Verify that changing loss_fun.lw affects the matching loss contribution."""
    print("--- test_loss_weight_mutation ---")
    params, scale, data, basal_mask, n_sub = _make_mock_setup()
    idxgall = list(range(n_sub))
    lw = [1.0, 1.0, 1.0, 1.0]
    eqn_all = (gov_eqn, front_eqn)
    solNN = solu_create(scale, basal_mask=basal_mask)

    loss_fun = loss_iso_create(solNN, eqn_all, scale, idxgall, lw, basal_mask=basal_mask)
    loss_fun.lref = 1.0  # don't normalize for this test

    # With matching weight = 1.0
    loss_fun.lw = jnp.array([1.0, 1.0, 1.0, 1.0])
    _, info_with = loss_fun(params, data)
    loss_md_with = info_with[0][4]

    # With matching weight = 0.0 (but matching error itself stays the same)
    loss_fun.lw = jnp.array([1.0, 1.0, 1.0, 0.0])
    loss_val_no_match, info_no = loss_fun(params, data)
    loss_md_no = info_no[0][4]
    total_loss_no = info_no[0][0]

    # The reported loss_md is the raw matching loss (before weighting),
    # but total_loss should differ because the matching contribution is zeroed
    # Verify by checking total loss
    loss_fun.lw = jnp.array([1.0, 1.0, 1.0, 1.0])
    _, info_with2 = loss_fun(params, data)
    total_loss_with = info_with2[0][0]

    if loss_md_with > 1e-10:
        assert total_loss_with > total_loss_no + 1e-10, \
            f"Total loss with matching ({total_loss_with}) should exceed without ({total_loss_no})"
        print(f"  loss_md = {loss_md_with:.6e}, total_with = {total_loss_with:.6e}, total_without = {total_loss_no:.6e}")
    else:
        print(f"  loss_md ~ 0 (trivial data), skipping magnitude check")

    print("  PASSED: loss weight mutation works correctly")


if __name__ == "__main__":
    test_build_grad_mask()
    test_gradient_masking()
    test_loss_weight_mutation()
    print("\n=== All staged training tests passed! ===")
