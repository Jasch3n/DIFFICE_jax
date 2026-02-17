
import jax
import jax.numpy as jnp
import optax
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffice_jax.optimizer.optimization import build_grad_mask

def reproduction():
    print("reproducing gradient mask issue...")
    
    # Mock parameters
    # Structure: {'net_u': [params1, None, params3], ...}
    # params1 is a dummy pytree
    
    params1 = {'w': jnp.ones((2, 2)), 'b': jnp.zeros((2,))}
    params3 = {'w': jnp.ones((2, 2)), 'b': jnp.zeros((2,))}
    
    params = {
        'net_u': [params1, None, params3],
        'net_mu': [params1, None, params3],
        'net_c': [None, None, None] # Maybe some are all None
    }
    
    # Mock basal mask: True for grounded, False for floating
    # Let's say region 0 is floating, region 1 is grounded (but None params anyway), region 2 is grounded
    basal_mask = [False, True, True]
    
    print("Basal Mask:", basal_mask)
    print("Params structure created.")
    
    # Test 1: build_grad_mask for Stage 1 (freeze grounded)
    # Grounded regions (1 and 2) should be frozen (mask 0.0), Floating (0) should be trainable (mask 1.0)
    print("\n--- Testing build_grad_mask (freeze_grounded=True) ---")
    mask_stage1 = build_grad_mask(params, basal_mask, freeze_grounded=True)
    
    # Check region 0 (Floating) -> Should be 1.0
    print("Region 0 (Floating) Mask w:", mask_stage1['net_u'][0]['w'])
    
    # Check region 2 (Grounded) -> Should be 0.0
    print("Region 2 (Grounded) Mask w:", mask_stage1['net_u'][2]['w'])
    
    # Check region 1 (Grounded, None) -> Should be None
    print("Region 1 (Grounded, None) Mask:", mask_stage1['net_u'][1])


    # Test 2: Simulate gradient masking application
    # This mimics _masked_minimizer logic: grads = tree_map(lambda g, m: g * m, grads, mask)
    print("\n--- Testing gradient masking application ---")
    
    # Create mock gradients (same structure as params)
    grads = {
        'net_u': [{'w': jnp.full((2,2), 0.5), 'b': jnp.full((2,), 0.5)}, None, {'w': jnp.full((2,2), 0.5), 'b': jnp.full((2,), 0.5)}],
        'net_mu': [{'w': jnp.full((2,2), 0.5), 'b': jnp.full((2,), 0.5)}, None, {'w': jnp.full((2,2), 0.5), 'b': jnp.full((2,), 0.5)}],
        'net_c': [None, None, None]
    }
    
    try:
        masked_grads = jax.tree_util.tree_map(lambda g, m: g * m if g is not None and m is not None else None, grads, mask_stage1)
        print("Gradient masking tree_map success (with manual check)!")
        print("Masked Grad Region 0 (Floating) w:", masked_grads['net_u'][0]['w'])
        print("Masked Grad Region 2 (Grounded) w:", masked_grads['net_u'][2]['w'])

    except Exception as e:
        print(f"Gradient masking tree_map FAILED with manual check: {e}")

    print("\n--- Testing gradient masking application (RAW - as in code) ---")
    # The actual code uses: lambda g, m: g * m
    try:
        # Note: tree_map might skip Nones if they are not leaves or if structures match structure
        # But if 'None' is a leaf in one and structure matches...
        
        # Let's see what happens if we just use the lambda from the code
        masked_grads_raw = jax.tree_util.tree_map(lambda g, m: g * m, grads, mask_stage1)
        
        print("Gradient masking tree_map (RAW) success!")
        # If successful, check values
        # We assume jax.tree_util.tree_map handles the None values correctly if structure matches
        # But if g is None and m is None, does it call the lambda?
        # Usually tree_map applies fn to leaves. None is a leaf (if not a pytree node).
        # If g is None and m is None, lambda(None, None) -> None * None -> Error!
        
    except Exception as e:
        print(f"Gradient masking tree_map (RAW) FAILED: {e}")
        
if __name__ == "__main__":
    reproduction()
