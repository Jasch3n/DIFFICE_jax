# Walkthrough - XPINN Basal Inversion Implementation (02/12/2026)

This document details the changes made to the codebase to enable simultaneous basal inversion of floating and grounded ice regions using the XPINN framework.

## Summary
The goal was to extend the XPINN framework to support mixed domains where some subregions represent floating ice (viscosity inversion) and others represent grounded ice (viscosity and basal friction inversion). This required updates to data preprocessing, network initialization, forward pass logic, and loss function construction.

## Detailed Changes

### 1. Data Preprocessing
File: `diffice_jax/data/xpinns/preprocessing.py`

- **Function `normalize_each`**:
    - Added `basal=False` argument.
    - Added logic to extract surface elevation `s` and boundary conditions (`bd_mu`, `bd_u`, `bd_v`) when `basal=True`.
    - Added normalization for these new variables.
    - Updated return signature to include specific boundary data for basal regions.

- **Function `normalize_data`**:
    - Added `basal_mask=None` argument.
    - Updated the `tree_map` call to pass the `basal` flag to `normalize_each` for each subregion based on the mask.

### 2. Model Initialization
File: `diffice_jax/model/xpinns/initialization.py`

- **Function `init_nets`**:
    - Added `basal_mask=None` argument.
    - Modified the initialization loop to check `basal_mask` for each subregion.
    - **Floating Regions (`basal=False`)**:
        - `net_u`: initialized with output size 3 (u, v, h).
        - `net_mu`: initialized.
        - `net_c`: set to `None`.
    - **Basal Regions (`basal=True`)**:
        - `net_u`: initialized with output size 4 (u, v, h, s).
        - `net_mu`: initialized.
        - `net_c`: initialized (new network for basal friction).
    - Updated return dictionary to include `net_c`.

### 3. Model Networks
File: `diffice_jax/model/xpinns/networks.py`

- **Function `solu_create`**:
    - Added `basal_mask=None` argument.
    - Updated the forward pass function `f`:
        - Checks `basal_mask[idx]`.
        - If **Floating**: Returns `[u, v, h, exp(mu)]`.
        - If **Basal**: Computes `c` from `net_c`, returns `[u, v, h, s, exp(mu), exp(c)]`.

### 4. Model Loss
File: `diffice_jax/model/xpinns/loss.py`

- **Helper Functions**:
    - Added/Restored `u_mag`, `nthrt`, and `sub_scale` helper functions.

- **Function `loss_iso_create`**:
    - Added `basal_mask=None` argument.
    - **Inner Function `loss_sub`**:
        - Added logic to handle `basal=True`.
        - **Basal Loss**: Added surface elevation mismatch, log-velocity mismatch, and boundary data fit (for `mu` at grounding line).
        - Constructed `err_all` vector to be consistent across region types (using zero-padding for unused slots), ensuring compatibility with `jnp.mean`.
    - **Inner Function `loss_match`**:
        - Updated to handle variable output indices.
        - `mu` index is 3 for floating regions, 4 for basal regions.
        - Correctly stitches `u, v, h, mu` across the interface regardless of region type.

## Verification

A verification script `tests/test_xpinn_basal.py` was created to test the implementation.
- **Scenario**: 2 subregions (Region 0: Floating, Region 1: Basal).
- **Process**:
    1. Initialized networks with mixed mask.
    2. Created mock data for both regions.
    3. Constructed loss function.
    4. Computed loss forward pass.
- **Result**: The script ran successfully, computing a valid loss value and confirming that all components (initialization, forward pass, loss calculation) are compatible.

```
Testing XPINN Basal Inversion...
Initializing networks...
Params structure verified.
Creating loss function...
Computing loss...
Loss computed: 1.047912359237671
Loss info shape: (16,)
Test passed!
```
