# DIFFICE_jax

DIFFICE_jax is a scientific computing context for differentiable neural-network data assimilation of ice flow. It uses PINNs and XPINNs to infer physical fields from remote-sensing or synthetic data while constraining the inference with ice-flow equations and boundary or interface conditions.

## Language

### Core Workflow

**Data Assimilation**:
Fitting neural-network fields to observed or synthetic ice-flow data while enforcing physical equations and constraints.
_Avoid_: Plain interpolation, curve fitting

**Inversion**:
Estimating latent physical fields from observed fields and governing equations. In this context the latent fields are usually effective viscosity, basal friction, or both.
_Avoid_: Prediction when the meaning is parameter estimation

**Joint Inversion**:
A coupled XPINN inversion that estimates effective viscosity and basal friction across interacting grounded and floating regions.
_Avoid_: Coupled training when the inferred fields are not both part of the objective

**Regression Workflow**:
A synthetic-data workflow where ground-truth latent fields are available for validation or direct supervision.
_Avoid_: Regression when it could mean statistical regression

**Ground Truth**:
Synthetic target values that are normally unavailable in real-data inversion, especially effective viscosity and basal friction.
_Avoid_: Truth for observed remote-sensing data

### Ice Domains

**Floating Region**:
A sub-region where ice is afloat and basal drag is absent. Floating regions use ice-shelf SSA physics and may use a calving-front dynamic boundary condition.
_Avoid_: Non-basal region, shelf region

**Grounded Region**:
A sub-region where ice is in contact with the bed and basal friction contributes to the momentum balance.
_Avoid_: Basal region, stream region

**Ice Shelf**:
A floating ice body whose motion is modeled as depth-integrated horizontal flow.
_Avoid_: Shelf when the grounded/floating distinction matters

**Ice Stream**:
A fast-flowing grounded ice body that may be coupled to an ice shelf at a grounding line.
_Avoid_: Grounded shelf

**Grounding Line**:
The physical transition where grounded ice begins to float. In an XPINN decomposition it is represented by an interface between a grounded region and a floating region.
_Avoid_: Interface when referring specifically to the physical transition

**Calving Front**:
The floating ice-shelf boundary where extensional stress balances ocean hydrostatic pressure.
_Avoid_: Boundary when the calving-front stress condition is meant

### Model Families

**PINN**:
A single-domain physics-informed neural network model family.
_Avoid_: Standalone XPINN

**Standalone PINN**:
A PINN used without XPINN domain decomposition, typically for one ice shelf or one grounded region.
_Avoid_: Regular PINN when contrasting with XPINN

**XPINN**:
An extended PINN model family that decomposes the domain into sub-regions and trains them with interface matching.
_Avoid_: Multi-PINN, partitioned PINN

**Sub-Region**:
One XPINN domain partition with its own fields, scale, sampled points, equation residuals, and optional boundary or interface constraints.
_Avoid_: Subdomain, block, tile

**Interface**:
The shared boundary between adjacent XPINN sub-regions. Use "grounding line" only when the interface is specifically the physical grounded/floating transition.
_Avoid_: Matching boundary in prose

**Paired Interface Coordinates**:
An interface point representation that stores coordinates from both adjacent sub-regions for one physical interface sample.
_Avoid_: Matching coordinates unless discussing legacy data keys

### Physical Fields And Notation

**Spatial Coordinates**:
The horizontal coordinates `(x, y)` in the ice-flow domain.
_Avoid_: Position when a coordinate pair is meant

**Velocity**:
The horizontal ice velocity field with components `(u, v)`.
_Avoid_: Speed when component direction matters

**Thickness**:
Ice thickness, denoted `h`.
_Avoid_: Height

**Surface Elevation**:
Ice surface elevation, denoted `s`. Floating-region surface elevation may be derived from flotation, while grounded-region surface elevation is an independent field.
_Avoid_: Surface when elevation is meant

**Effective Viscosity**:
The isotropic viscosity field inferred by the ice-flow inversion, denoted `mu`.
_Avoid_: Ice viscosity when anisotropic components are being discussed

**Horizontal Viscosity**:
The horizontal component of anisotropic viscosity, denoted `mu_h`.
_Avoid_: Effective viscosity

**Vertical Viscosity**:
The vertical component of anisotropic viscosity, denoted `mu_v`.
_Avoid_: Effective viscosity

**Basal Friction**:
The grounded-region basal resistance field, denoted `C` when represented as a coefficient.
_Avoid_: Basal drag coefficient unless the equation derivation specifically uses drag

**Floating Rheology Parameter**:
A planned floating-region learned field for future physics that needs a quantity beyond velocity, thickness, surface elevation, and effective viscosity.
_Avoid_: Extra viscosity

**Strain Rate**:
A velocity-gradient magnitude used in equation diagnostics and regularization reasoning.
_Avoid_: Rate without specifying the differentiated field

### Equations And Constraints

**SSA Equation**:
The shallow-shelf approximation momentum-balance equation. Floating SSA omits basal friction; grounded SSA includes driving stress from surface slope and basal friction.
_Avoid_: Stokes equation when referring to the implemented depth-integrated balance

**Isotropic SSA**:
An SSA equation using one effective viscosity field.
_Avoid_: SSA when the anisotropic distinction matters

**Anisotropic SSA**:
An SSA equation using separate horizontal and vertical viscosity components.
_Avoid_: Isotropic SSA

**Dynamic Boundary Condition**:
The stress-balance condition at the calving front of a floating region.
_Avoid_: Boundary loss when naming the physical condition

**Equation Residual**:
The difference between the two sides of a governing equation after all terms are expressed in the chosen normalized or physical units.
_Avoid_: Error when the quantity is a PDE residual

**Data Loss**:
The loss term that fits model outputs to observed or synthetic target fields.
_Avoid_: Observation loss

**Equation Loss**:
The loss term that penalizes equation residuals at collocation points.
_Avoid_: Physics loss when a more specific loss term is available

**Calving-Front Loss**:
The loss term associated with the dynamic boundary condition at a floating calving front.
_Avoid_: Boundary loss when the boundary is specifically the calving front

**Matching Loss**:
The XPINN loss term that enforces continuity or controlled transfer across interfaces between adjacent sub-regions.
_Avoid_: Continuity loss except when referring to the mathematical concept in published XPINN descriptions

**Component Weights**:
Weights among residual components inside one loss term, such as velocity versus thickness or x-momentum versus y-momentum.
_Avoid_: Global weights

**Global Weights**:
Weights that multiply whole loss terms before they contribute to a scalar loss or residual-vector objective.
_Avoid_: Family weights, component weights

**C0 Matching**:
Zeroth-order interface matching of predicted fields.
_Avoid_: Value matching when contrasting with C1 matching

**C1 Matching**:
First-derivative interface matching, and selected higher-order matching when explicitly defined by the workflow.
_Avoid_: Gradient matching when the workflow includes additional derivative terms

**Symmetric Interface Viscosity Matching**:
The real-data grounding-line coupling where adjacent grounded and floating regions are trained simultaneously through matching loss.
_Avoid_: One-way viscosity transfer

**Grounded Interface Viscosity Boundary**:
A legacy synthetic or regression condition that supplies grounded-side interface viscosity from an adjacent floating region.
_Avoid_: Symmetric interface viscosity matching

**gPINN Regularization**:
A gradient-enhanced PINN term that penalizes gradients of equation residuals.
_Avoid_: Gradient regularization when the gradient is specifically of the residual

**Mu-Gradient Regularization**:
A regularization term on the spatial gradient of effective viscosity.
_Avoid_: gPINN regularization

### Sampling, Scaling, And Data

**Collocation Points**:
Points where equation residuals are evaluated.
_Avoid_: Data points

**Interface Collocation**:
Additional collocation points sampled near grounded/floating interfaces so equation residuals are evaluated near the cross-domain transition.
_Avoid_: Interface points when the points are used for equation residuals rather than matching

**RAD Sampling**:
Residual-based adaptive distribution sampling that biases collocation sampling toward larger equation residuals.
_Avoid_: Adaptive sampling when the residual-based method is specifically meant

**Normalized Data**:
Data transformed to dimensionless neural-network coordinates and outputs using per-region centering, ranges, and physical scales.
_Avoid_: Standardized data unless only mean-variance scaling is meant

**Physical Units**:
The dimensional coordinate, field, residual, or diagnostic values before normalization or after redimensionalization.
_Avoid_: Raw units when the values have already been processed but remain dimensional

**Redimensionalized Output**:
Network output converted from normalized representation back to physical units.
_Avoid_: Denormalized output when physical units are the important distinction

**Region Scale**:
The per-sub-region bundle of centering, range, and dynamic scales used for normalization and redimensionalization.
_Avoid_: Scale when it is unclear whether data ranges or dynamic scales are meant

**Dynamic Scale**:
The physical scales used to non-dimensionalize equation terms and physical fields, including length, velocity, viscosity, basal friction, and stress scales.
_Avoid_: Data range

**Basal Mask**:
The ordered grounded/floating classifier for XPINN sub-regions.
_Avoid_: Region kind when discussing the boolean representation

**Raw MATLAB Dataset**:
Input `.mat` data in the expected PINN or XPINN layout.
_Avoid_: Processed data

**Processed Data Output**:
The normalized data and metadata produced by preprocessing.
_Avoid_: Raw dataset

**Sample Batch**:
The per-iteration sampled collection of data, collocation, boundary, and interface points passed to a loss function.
_Avoid_: Dataset

**Canonical Internal Name**:
A preferred name used in new internal code for a domain concept.
_Avoid_: Legacy alias in new internal modules

**Legacy Alias**:
An existing public API name, dataset key, or test-facing name preserved at boundaries for compatibility.
_Avoid_: Canonical name

**Stitched Prediction**:
An XPINN prediction reconstructed from sub-region outputs onto a whole-domain array.
_Avoid_: Merged output when the reconstruction is spatially indexed

### Training And Diagnostics

**Public Workflow**:
A user-facing workflow category such as ice-shelf-only, joint-inversion, or joint-inversion-regression.
_Avoid_: Model family

**Model Family**:
The neural-network decomposition family, either PINN or XPINN.
_Avoid_: Public workflow

**Training Stage**:
One ordered optimizer phase in a training run.
_Avoid_: Epoch

**Adam Stage**:
The stochastic-gradient training stage using Adam.
_Avoid_: Adam optimizer when the phase of training is meant

**L-BFGS Stage**:
The quasi-Newton training stage using L-BFGS.
_Avoid_: LBFGS without the hyphen in prose

**KFAC Stage**:
The approximate second-order training stage using KFAC.
_Avoid_: K-FAC in prose unless matching an external package name

**Scalar Loss**:
The diagnostic weighted sum of mean-squared loss terms.
_Avoid_: Objective when the optimized objective is a residual vector

**KFAC Residual Vector**:
The concatenated weighted residual vector used to define the least-squares objective optimized by KFAC.
_Avoid_: Scalar loss

**KFAC Objective**:
The least-squares objective formed from the KFAC residual vector.
_Avoid_: Loss history when referring to the optimized scalar

**Active Regions**:
The subset of XPINN sub-regions included in a loss or optimizer step.
_Avoid_: Selected domains

**Freeze Spec**:
A parameter-freezing description that keeps selected region or network parameters fixed during training.
_Avoid_: Frozen params when referring to the specification rather than the parameters

**First Stage**:
The initial regression-training stage that fits the main XPINN fields.
_Avoid_: Stage 1 unless naming a file or artifact

**Second Stage**:
The follow-up regression-training stage that fits a correction target from first-stage residuals.
_Avoid_: Stage 2 unless naming a file or artifact

**Transect Diagnostic**:
A plot or sample along a fixed x or y line used to inspect predicted fields, stresses, gradients, residuals, or derivatives.
_Avoid_: Profile when the diagnostic is spatially sampled through the domain

## Notation

**`x`, `y`**:
Horizontal spatial coordinates.
_Avoid_: `X` for a single coordinate pair

**`u`, `v`**:
Horizontal velocity components in the x and y directions.
_Avoid_: `speed` for component fields

**`h`**:
Ice thickness.
_Avoid_: `H` in prose unless quoting a legacy data structure

**`s`**:
Surface elevation.
_Avoid_: `surface` as a symbol

**`mu`**:
Effective isotropic viscosity.
_Avoid_: `eta` unless an external derivation requires it

**`mu_h`, `mu_v`**:
Horizontal and vertical anisotropic viscosity components.
_Avoid_: `mu1`, `mu2`

**`C`**:
Basal friction coefficient.
_Avoid_: `c0` for the physical field; `c0` is a scale

**`rho`, `rho_w`, `g`**:
Ice density, ocean-water density, and gravitational acceleration.
_Avoid_: Unqualified density

**`l0`, `u0`, `mu0`, `c0`, `term0`**:
Dynamic scales for length, velocity, effective viscosity, basal friction, and stress-like equation terms.
_Avoid_: Using these names for physical fields

**`gamma_mu`, `gamma_c`**:
Dynamic-scale factors for viscosity and basal-friction scaling.
_Avoid_: Loss weights
