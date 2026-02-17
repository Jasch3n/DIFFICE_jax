# Basal Inversion Development Log

This document tracks the development of simultaneous basal inversion (viscosity + friction) for floating and grounded ice using PINNs and XPINNs.

---

## 02/12/2026 19:30 — XPINN Basal Inversion: Initial Implementation

Extended the XPINN framework to support mixed domains where some subregions represent floating ice (viscosity inversion) and others represent grounded ice (viscosity + basal friction inversion).

### Data Preprocessing (`diffice_jax/data/xpinns/preprocessing.py`)

- **`normalize_each`**: Added `basal=False` argument. When `basal=True`, extracts and normalizes surface elevation `s` and boundary conditions (`bd_mu`, `bd_u`, `bd_v`). Updated return signature.
- **`normalize_data`**: Added `basal_mask=None` argument. Passes the `basal` flag per subregion to `normalize_each`.

### Model Initialization (`diffice_jax/model/xpinns/initialization.py`)

- **`init_nets`**: Added `basal_mask=None` argument. Per-subregion initialization:
  - **Floating**: `net_u` → 3 outputs (u, v, h). `net_c` → `None`.
  - **Grounded**: `net_u` → 4 outputs (u, v, h, s). `net_c` → initialized for basal friction.

### Model Networks (`diffice_jax/model/xpinns/networks.py`)

- **`solu_create`**: Added `basal_mask=None`. Forward pass checks `basal_mask[idx]`:
  - Floating → `[u, v, h, exp(mu)]`
  - Grounded → `[u, v, h, s, exp(mu), exp(c)]`

### Model Loss (`diffice_jax/model/xpinns/loss.py`)

- **`loss_iso_create`**: Added `basal_mask=None`.
  - `loss_sub`: Handles `basal=True` with surface elevation mismatch, log-velocity mismatch, and grounding-line mu boundary data. Constructs `err_all` with zero-padding for consistent shape.
  - `loss_match`: Variable `mu` column index (3 for floating, 4 for grounded). C0/C1/C2 stitching across mixed interfaces.

### Verification

Created `tests/test_xpinn_basal.py` — 2 subregions (floating + grounded), verifies initialization, forward pass, and loss computation.

```
Loss computed: 1.047912359237671
Loss info shape: (16,)
Test passed!
```

---

## 02/13/2026 21:50 — XPINN Grounded Component: Troubleshooting & Bug Fixes

The XPINN ran successfully but produced very low friction at the pinning point (a grounded region surrounded by floating ice). Systematic comparison against the successful regular PINN implementation revealed 5 bugs.

### Bug 1: Viscosity scale `mu0` used wrong gravity (~9x error)

**Files**: `loss.py`, `prediction.py`

`sub_scale()` and `extract_scale()` always used reduced gravity `gd = g*(1-ρ/ρ_w) ≈ 1.07` for `mu0`. For grounded ice, PINN correctly uses full gravity `g = 9.8`. This ~9x error in viscosity scale directly corrupted the momentum balance and friction inference.

**Fix**: Both functions now accept a `basal` flag and use `g_eff = g if basal else gd`.

```diff
-    gd = 9.8 * (1 - rho / rho_w)
-    mu0 = rho * gd * h0 * (l0m / u0m)
+    g = 9.8
+    gd = g * (1 - rho / rho_w)
+    g_eff = g if basal else gd
+    mu0 = rho * g_eff * h0 * (l0m / u0m)
```

### Bug 2: `prediction.py` read `mu` from wrong column for grounded regions

**File**: `prediction.py`

Grounded network output is `[u, v, h, s, mu, c]` — `mu` is at column 4. Code read column 3 (which is surface elevation `s`), making all post-training viscosity visualizations for grounded regions incorrect.

**Fix**: `redimensionalize()` now selects `mu_col = 4 if basal else 3`.

### Bug 3: `net_output` didn't pass `basal=True` to `gov_eqn`

**File**: `prediction.py`

At prediction time, `gov_eqn` was called without `basal=True`, so equation residue diagnostics used floating SSA equations even for grounded subregions.

**Fix**: `net_output()` now accepts `basal` flag, passed through from `predict()`.

### Bug 4: Friction scale `c0` missing from XPINN prediction

**File**: `prediction.py`

PINN computes `c0 = (h0 * mu0) / l0m²` for dimensionalizing basal friction. XPINN `extract_scale()` did not compute `c0`, so `beta` output was in raw network units.

**Fix**: `extract_scale()` now computes and returns `c0`. `beta` is read from column 5 (not 4) and multiplied by `c0`.

### Bug 5: `data_log_u_err` computed but never used in loss

**File**: `loss.py`

PINN includes log-velocity error (`log(|u_pred|/|u_smp|)`) in the grounded loss, which helps with low-velocity regions near the pinning point. XPINN computed this value but never added it to `data_err_all`.

**Fix**: `data_log_u_err` now added to `data_err_all` with weight 0.6. Error vector size increased from 8 to 9; weight and index slicing updated accordingly.

### Additional: `infer_xpinn.py`

Updated `predict_xpinn(...)` call to pass `basal_mask=basal_mask` so the prediction pipeline applies all the above fixes.

### Verification

`test_xpinn_basal.py` updated and passes:

```
Loss computed: 76.98200225830078
Loss info shape: (17,)
Initial matching loss: 7.027885437011719
Perturbed matching loss: 5.2337236404418945
SUCCESS: Matching loss changed when viscosity parameters were perturbed.
Test passed!
```

### Next Step

Re-run full XPINN training on pinning point data to verify improved friction prediction.

---

## 02/14/2026 20:45 — Two-Stage Adam Training for Basal Inversion

Added `basal_twoStage_adam_optimizer` to support sequential training of floating and grounded subregions, addressing the pinning-point scenario where the floating viscosity should be learned first, then the grounded friction can bootstrap off the interface matching conditions.

### Optimizer (`diffice_jax/optimizer/optimization.py`)

- **`build_grad_mask(params, basal_mask, freeze_grounded)`**: Creates a pytree of 0/1 masks matching the parameter structure. When `freeze_grounded=True`, all grounded subregion params are masked to 0 (frozen); vice versa.
- **`basal_twoStage_adam_optimizer(...)`**: Runs two Adam stages:
  - **Stage 1**: Floating regions train (grounded frozen), matching loss excluded via `stage1_lw`.
  - **Stage 2**: Grounded regions train (floating frozen), matching loss included via `stage2_lw`.
  - Each stage re-initializes optimizer state and re-computes `lref` for loss normalization.
  - Supports RAD adaptive sampling within each stage.

### Loss Function (`diffice_jax/model/xpinns/loss.py`)

- `loss_fun.lw` is now a **mutable attribute** instead of captured from the closure. This allows the two-stage optimizer to swap loss weights between stages without recreating the loss function (avoiding re-JIT).

### Inference Script (`PinningPointInversion/Scripts/infer_xpinn.py`)

- Parses new `staged_training` JSON config block:
  ```json
  "staged_training": {
      "enabled": true,
      "stage1_epochs": 5000,
      "stage2_epochs": 5000,
      "stage1_loss_weights": [1.0, 0.5, 0.1, 0.0],
      "stage2_loss_weights": [1.0, 0.5, 0.1, 0.5]
  }
  ```
- Conditionally calls `basal_twoStage_adam_opt` when enabled; falls back to original `adam_opt` otherwise.

### Public API (`diffice_jax/__init__.py`)

- Exported `basal_twoStage_adam_optimizer` as `basal_twoStage_adam_opt`.

### Verification

Created `tests/test_staged_training.py` — 3 tests:

```
--- test_build_grad_mask ---
  PASSED: gradient masks are correct
--- test_gradient_masking ---
  PASSED: gradient masking correctly freezes/unfreezes params
--- test_loss_weight_mutation ---
  loss_md = 6.958958e+00, total_with = 8.889511e+00, total_without = 1.930553e+00
  PASSED: loss weight mutation works correctly
=== All staged training tests passed! ===
```

Existing `test_xpinn_basal.py` regression test also passes.
