# Ubiquitous Language

This glossary records the project terms to use in code, docs, tests, analysis, and handoffs. Prefer these names unless a legacy public API, MATLAB data key, or published mathematical convention requires an existing name.

## Core Domain

### DIFFICE_jax
The JAX package for differentiable neural-network data assimilation of ice flow. It uses PINNs and XPINNs to infer physical fields from remote-sensing or synthetic data.

### Data Assimilation
The workflow that fits neural networks to observed velocity, thickness, and related fields while constraining them with ice-flow equations and boundary/interface conditions.

### Inversion
Estimating latent physical fields from observations and equations. In current code this usually means viscosity inversion for floating ice shelves, basal friction inversion for grounded ice streams, or joint inversion of both fields across coupled regions.

### Joint Inversion
A coupled XPINN workflow that simultaneously estimates ice viscosity and basal friction. The near-term target is the two-region joint inversion described in `docs/todos_20260611.md`: one grounded ice-stream region coupled to one floating ice-shelf region.

### Regression Workflow
The synthetic-data development and validation workflow where ground-truth viscosity and basal friction are available. In code this appears as `use_regression`, `loss_regression_create`, `data_regression_sample_create`, and scripts under `tests/test_xpinn_regression`.

Use "regression workflow" for this hard-coded synthetic workflow. Avoid using "regression" alone when it could be confused with statistical regression.

## Ice Domains

### Floating Region
An ice-shelf sub-region. It has no basal drag term, uses floating SSA equations, and may have a calving-front dynamic boundary condition. The current vanilla floating SSA workflow derives surface elevation from thickness by flotation, but future floating SSA variants may predict surface elevation directly.

Code aliases: `basal=False`, `basal_mask[idx] == False`.

### Grounded Region
An ice-stream or grounded-ice sub-region. It includes basal friction, predicts surface elevation as a network output, and uses a grounded SSA-style equation with driving stress from surface slope and basal drag.

Code aliases: `basal=True`, `basal_mask[idx] == True`.

### Ice Shelf
A floating ice body. Historically the main target of DIFFICE_jax and current PINN/XPINN docs.

### Ice Stream
A grounded fast-flowing ice body. In the current development plan it is coupled to an ice shelf at the grounding line for basal friction inversion.

### Grounding Line
The physical transition where grounded ice begins to float and becomes an ice shelf. In two-region joint inversion it is represented as an XPINN interface between a grounded region and a floating region.

Preferred abbreviation: `GL` only after first spelling out "grounding line".

### Calving Front
The floating ice-shelf boundary where dynamic boundary conditions balance extensional stress against ocean hydrostatic pressure.

Code/data aliases: `xct`, `yct`, `nnct`, `ct`, `front_eqn`, `dbc_*`.

## Model Families

### PINN
A single-domain physics-informed neural network workflow. It can perform ice-shelf inversion, or grounded-only inversion when required boundary information such as grounding-line viscosity is supplied externally.

Use "standalone PINN" when contrasting with XPINNs.

### XPINN
An extended PINN workflow with domain decomposition into sub-regions and matching losses across interfaces. XPINNs are the current vehicle for coupled grounded/floating joint inversion.

Code aliases: `xpinns`, `idxgall`, `n_sub`, `n_subregions`.

### Sub-Region
One XPINN domain partition with its own networks, normalization scale, sampled batches, equation residuals, and optional boundary/interface constraints.

Use "sub-region" in prose. Existing code often uses `sub`, `idx`, `idxgall`, or "region".

### Interface
The shared boundary between adjacent XPINN sub-regions. Interface points are used for matching losses; additional nearby collocation points may be appended when adjacent regions have different grounded/floating masks.

Code/data aliases: `x_md`, `y_md`, `X_md`, `md`, `matching coordinates`.

### Paired Interface Coordinates
An interface batch where each row contains coordinates from both sides of the interface: left-region `(x, y)` and right-region `(x, y)`. This supports non-identical normalized coordinates on the two sides.

Code shape: `(n, 4)` with columns `[x_left, y_left, x_right, y_right]`.

## Physical Fields

### Velocity
Horizontal ice velocity components.

Code/data aliases: `u`, `v`, `ud`, `vd`, `U_smp`, `data_u_err`.

### Thickness
Ice thickness.

Code/data aliases: `h`, `hd`, `H_smp`, `data_h_err`.

### Surface Elevation
Ice surface elevation. Floating regions derive it from thickness by flotation; grounded regions use observed or predicted surface elevation.

Code/data aliases: `s`, `sd`, `S_smp`, `data_s_err`.

### Effective Viscosity
The inferred isotropic ice viscosity field.

Preferred symbol: `mu`.

Code/data aliases: `mu`, `mud`, `net_mu`, `Mu_smp`, `data_mu_err`. In regression losses, compare viscosity through `log(mu)` unless a specific test says otherwise.

### Floating Rheology Parameter
A planned additional floating-region field for future SSA variants where ice-shelf physics needs an extra learned variable beyond velocity, thickness, surface elevation, and effective viscosity.

Preferred symbol: `k`.

Code/data aliases: none yet.

### Basal Friction
The grounded-region basal resistance field. Current code often represents this with a positive neural-network output and a dynamic scale `c0`.

Preferred symbol in prose: `C` or "basal friction coefficient" when matching the existing data field.

Code/data aliases: `C`, `alpha2d`, `net_c0`, `C_smp`, `data_C_err`, `c0`.

### Horizontal Viscosity
The horizontal component in anisotropic viscosity inversion.

Preferred symbol: `mu_h`.

### Vertical Viscosity
The vertical component in anisotropic viscosity inversion.

Preferred symbol: `mu_v`.

### Strain Rate
A derived normalized velocity-gradient magnitude used in equation diagnostics and network gradient output.

Code aliases: `strate`.

## Equations And Constraints

### SSA Equation
The shallow-shelf approximation momentum-balance equation. Floating regions use the ice-shelf form; grounded regions add surface-slope driving stress and basal-friction terms.

Code aliases: `ssa_iso`, `ssa_aniso`, `gov_eqn`, `eqn`.

### Second-Stage Equation
The equation variant used by the second-stage regression workflow. For grounded regions it currently emphasizes a first-direction SIA-like residual and is separate from the main isotropic SSA equation.

Code aliases: `eqn_secondStage`, `loss_regression_2ndstage_create`.

### Dynamic Boundary Condition
The calving-front stress-balance condition for floating regions.

Code aliases: `front_eqn`, `dbc_iso`, `dbc_aniso`, `ct` loss.

### Data Loss
The loss term that fits neural network outputs to observed or synthetic target fields: velocity, thickness, surface elevation, viscosity, and basal friction depending on workflow.

Code aliases: `loss_data`, `data_err`, `data_w`.

### Equation Loss
The loss term that penalizes PDE residuals at collocation points.

Code aliases: `loss_eqn`, `eqn_err`, `eqn_w`, `gamma_eq`, `eqn_region_weights`.

### Calving-Front Loss
The boundary-condition loss on calving-front points for floating regions, or the special grounded interface viscosity boundary loss when `grounded_only_interface_mu_ct` is enabled.

Code aliases: `loss_ct`, `ct_err`, `ct_w`.

### Matching Loss
The XPINN interface loss that enforces continuity or controlled transfer across adjacent sub-regions.

Code aliases: `loss_md`, `md_err`, `match`, `match_weight`, `match_component_weights`.

### Component Weights
Relative weights among residual components inside one loss term. Use this term for weights such as `data_w`, `eqn_w`, `ct_w`, and `md_w`: these choose how velocity versus thickness, x-equation versus y-equation, calving-front components, or individual matching components are weighted within their own term.

Code aliases: `data_w`, `eqn_w`, `ct_w`, `md_w`, `match_component_weights`.

### Global Weights
Weights that multiply an entire loss term before it contributes to `scalar_loss` or to the KFAC residual objective. Use this term for whole-term multipliers such as the data, equation, calving-front, matching, gPINN, and mu-gradient weights. Do not call these "family weights."

Code aliases: `data_global_weight`, `eqn_global_weight`, `ct_global_weight`, `match_weight`, `gpinn_weight`, `mu_grad_weight`.

### C0 Matching
Zeroth-order interface matching of predicted fields. In current regression matching this includes velocity, thickness, surface elevation, and `log(mu)`.

Code aliases: `C0_res`.

### C1 Matching
First- and selected higher-order interface matching. In current regression matching this includes gradients of velocity/thickness/surface, gradients of `log(mu)`, and selected strain-rate derivative terms.

Code aliases: `C1_res`.

### Grounded Interface Viscosity Boundary
A legacy synthetic/regression condition that supplies viscosity on the grounded side of a grounded/floating interface from the adjacent floating region. This is not part of the target real-data joint inversion workflow.

Code alias: `grounded_only_interface_mu_ct`.

### Symmetric Interface Viscosity Matching
The target real-data grounding-line viscosity coupling: adjacent grounded and floating regions are trained simultaneously through matching loss at the interface, while their data losses and equation losses remain region-specific.

Use this term for new code. Treat one-way floating-to-grounded viscosity transfer as legacy regression-workflow behavior only.

### gPINN Regularization
A gradient-enhanced PINN term that penalizes gradients of the equation residuals.

Code aliases: `gpinn_weight`, `gpinn_res`, `loss_gpinn`.

### Mu-Gradient Regularization
A floating-region regularization term on the spatial gradient of viscosity.

Code aliases: `mu_grad_weight`, `mu_grad_res`, `loss_mu_grad`.

## Sampling And Scaling

### Collocation Points
Points where equation residuals are evaluated.

Code/data aliases: `xcol`, `ycol`, `X_col`, `col`.

### Interface Collocation
Additional collocation points sampled near grounded/floating interfaces so equation residuals see the cross-domain transition.

Code aliases: `interface_collocation_libraries`, `N_INTERFACE_LIBRARY`, `N_INTERFACE_COLLOCATION`.

### RAD Sampling
Residual-based adaptive distribution sampling that biases collocation sampling by equation residual magnitude.

Code aliases: `eval_RAD_probs`, `eval_adaptive`, `adaptive_probs`.

### Normalized Data
Data transformed to dimensionless neural-network coordinates and outputs using per-region means, ranges, and dynamic scales.

Code aliases: `normalize_data`, `normalize_each`, `data_norm`.

### Redimensionalized Output
Network predictions converted back to physical units for evaluation, plotting, or stitching.

Code aliases: `redimensionalize`, `predict`.

### Region Scale
The per-sub-region normalization and physical scaling bundle.

Code types: `SubScaleResult`, containing `DataMean`, `DataRange`, and `DynamicScale`.

### Dynamic Scale
The physical scales used to non-dimensionalize equation terms and outputs, including length, velocity, viscosity, basal friction, and stress scales.

Code type: `DynamicScale`; fields include `l0`, `u0`, `mu0`, `c0`, `term0`, `gamma_mu`, and `gamma_c`.

### Basal Mask
The ordered grounded/floating classifier for XPINN sub-regions.

Code alias: `basal_mask`; `True` means grounded, `False` means floating.

## Training And Optimization

### Adam Stage
The stochastic gradient optimization stage.

Code aliases: `adam_optimizer`, `adam_opt`.

### L-BFGS Stage
The quasi-Newton optimization stage.

Code aliases: `lbfgs_optimizer`, `lbfgs_opt`.

### KFAC Stage
The KFAC optimizer stage used by regression experiments and residual-vector objectives when available.

Code aliases: `KfacOptimizer`, `kfac_optimize`, `kfac_residuals`, `kfac_objective`.

### KFAC Residual Vector
The concatenated, weighted residual vector used to define the least-squares objective for KFAC.

Code aliases: `kfac_residuals`, `residual_objective`.

### Scalar Loss
The diagnostic weighted sum of mean-squared loss terms. It uses component weights inside each term, then applies global weights to the data, equation, calving-front, matching, gPINN, and mu-gradient terms.

Code aliases: `scalar_loss`, `loss_info[0]`.

### KFAC Objective
The least-squares objective optimized by KFAC. It squares and sums the weighted KFAC residual vector, then normalizes by `lossf.lref`. The same component weights and global weights are represented as square-root residual scales before the residuals are concatenated.

Code aliases: `objective`, `loss_n`, `kfac_objective`.

### Region Term Weights
Per-region weights for balancing data, equation, boundary, matching, gPINN, and mu-gradient terms.

Code aliases: `region_term_weights`, `compute_region_term_weights`, `current_region_term_weight`.

### Active Regions
The subset of XPINN sub-regions currently included in the loss or optimizer step.

Code alias: `active_regions`.

### Freeze Spec
A tree-shaped parameter-freezing specification used by regression scripts to keep selected region/network parameters fixed.

Code aliases: `freeze_spec`, `freeze_mask`, `frozen_params`, `apply_frozen_params`.

### First Stage
The initial regression-training stage that fits the main XPINN fields and produces velocity residuals for a later correction stage.

Code aliases: `params_1st`, `first_stage_params`, `first_stage_velocity_misfit`.

### Second Stage
The follow-up regression-training stage that fits a velocity-only correction target from the first-stage velocity residual.

Code aliases: `loss_regression_2ndstage_create`, `params_2nd`.

## Data Layout

### Raw MATLAB Dataset
Input `.mat` data in either PINN or XPINN layout. XPINN datasets use MATLAB cell arrays per sub-region.

Important keys: `xd`, `yd`, `ud`, `vd`, `xd_h`, `yd_h`, `hd`, `sd`, `xct`, `yct`, `nnct`, `x_md`, `y_md`, `xcol`, `ycol`, `mud`, `alpha2d`.

### Processed Data Output
The normalized data and metadata returned by preprocessing or wrapped by regression scripts.

Code aliases: `data_all`, `data_output`, `DataOutput`.

### Sample Batch
The per-iteration sampled data dictionary used by loss functions.

Code aliases: `data`, `dataf`, `DataSample`; main keys are `smp`, `col`, `ct`, and `md`.

### Canonical Internal Name
A preferred name used inside new concept modules.

Examples: `interface` instead of `md`, `calving_front` instead of `ct`, `sub_region_indices` instead of `idxgall`, and `basal_friction` instead of `alpha2d`.

### Legacy Alias
An existing public API name, MATLAB dataset key, or test-facing name preserved at data-loading and public-wrapper boundaries for compatibility.

### Stitched Prediction
An XPINN prediction reconstructed from sub-region outputs onto whole-domain arrays.

Code aliases: `stitch`, `idxcrop`, `idxcrop_h`.

## Diagnostics And Experiments

### Hard-Coded Feature
A behavior in regression scripts/tests that is specialized to the current synthetic examples rather than part of a general DIFFICE_jax API.

Use this term when auditing `tests/test_xpinn_regression`.

### Ground Truth
Synthetic target values that are normally unavailable for real data, especially viscosity `mud` and basal friction `alpha2d`.

### Characteristic Reduction
The diagnostic method in `tests/test_xpinn_regression/calc_char.py` that computes characteristic curves and recovers basal friction-related fields from grounded-region kinematics.

Code aliases: `pq_terms`, `rg_terms`, `trace_characteristic`, `hnu`, `beta_grid`.

### Transect Diagnostic
A plot or sample along a fixed x or y line used to inspect predicted state, stress, viscosity gradients, residuals, or second derivatives.

Code aliases: `plot_stress_mu_grad_transect.py`, `transect_plots`.

## Naming Preferences

- Use `floating region` and `grounded region` in prose; use `basal_mask` only for the code-level boolean.
- Use `interface` for XPINN sub-region boundaries; use `grounding line` only for the physical grounded/floating transition.
- Use `matching loss` for XPINN interface penalties; reserve `continuity loss` for the mathematical PINN/XPINN concept from the published docs.
- Use `effective viscosity` for isotropic `mu`; use `horizontal viscosity` and `vertical viscosity` for anisotropic components.
- Use `basal friction` or `basal friction coefficient` for `C`/`alpha2d`; avoid introducing a second prose term for the same field unless the equation derivation requires it.
- Use `regression workflow` for synthetic ground-truth-assisted development; use `test` or `regression test` only for pytest artifacts.
- Use canonical internal names in new concept modules. Keep legacy aliases such as `ct`, `md`, `x_md`, `nnct`, `alpha2d`, and `idxgall` only where required by existing APIs, dataset schemas, or compatibility wrappers.
- Do not introduce one-way floating-to-grounded viscosity transfer in new real-data joint inversion code.
