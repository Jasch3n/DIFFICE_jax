# Multi-Stage Neural Networks (MSNN) for Physics-Informed Neural Networks (PINNs)
### Implementation Guide — Based on Wang & Lai (2024), *J. Comput. Phys.* 504, 112865

---

## Core Idea

Standard PINNs plateau around O(10⁻⁵) error because neural networks suffer from **spectral bias** — they learn low-frequency features easily but struggle with high-frequency residuals. The MSNN approach addresses this by training a sequence of networks, where each new network is optimized to fit the *residue* left by all previous stages. Over successive stages, accuracy can approach machine precision O(10⁻¹⁶).

---

## Overview of the Algorithm (Algorithm 4)

**Prerequisite:** Normalize the governing equation so the largest term is O(1).

### Stage 1 — Standard PINN Training

1. Build a neural network `u₀(x)` with regular weight initialization (e.g., Xavier).
2. Sample collocation points and boundary condition data points.
3. Train using the standard PINN loss:

$$\mathcal{L} = (1 - \gamma)\mathcal{L}_d + \gamma \mathcal{L}_e$$

where $\mathcal{L}_d$ is the data/boundary loss and $\mathcal{L}_e$ is the PDE residual loss.

4. After training, compute the **equation residue** `r₁(x, u₀)` by evaluating the PDE with the trained network.

---

### Stage k+1 — Higher-Stage Training

For each subsequent stage, three key quantities must be estimated from the previous stage's equation residue.

#### Step 1 — Estimate the Dominant Frequency of the New Network (`κ`)

The dominant frequency `f_d` of the next-stage network mirrors that of the equation residue:

$$f_d^{(1)} \approx f_d^{(r)}$$

where $f_d^{(r)}$ is the dominant frequency of the current equation residue. Compute this via FFT of the residue.

Use this to set the **scale factor** for the new network:

$$\hat{\kappa} > \pi f_d \quad \Rightarrow \quad \kappa = \hat{\kappa} / \sqrt{Var}$$

This scale factor multiplies the weights between the **input layer and first hidden layer only** to allow the network to capture high-frequency features. Also use **sin(x)** as the activation function in the first hidden layer (keep tanh for remaining layers).

#### Step 2 — Estimate the Magnitude Prefactor (`ε`)

For a PDE of order `m`, the magnitude of the prediction error can be estimated from the equation residue:

$$\epsilon_1 = \frac{\epsilon_{r_1}}{[2\pi f_d^{(r)}]^m \cdot \epsilon_\beta}$$

where:
- $\epsilon_{r_1} = \text{RMS}(r_1(x, u_0))$ is the RMS of the equation residue
- $\epsilon_\beta = \text{RMS}(\beta_m)$ is the RMS of the coefficient of the highest-order derivative term, evaluated at the previous network
- $m$ is the order of the highest derivative in the PDE

> **Tip:** For simple normalized equations where $\epsilon_\beta \approx 1$, this simplifies to $\epsilon_1 \approx \epsilon_{r_1} / (2\pi f_d)^m$.

Alternatively, use **Algorithm 2** (iterative magnitude estimation) for linear equations: define a guess solution proportional to the source function and iterate until the ratio of the differential operator applied to the guess vs. the source is in [0.1, 10].

#### Step 3 — Determine the Equation Weight (`γ`)

For high-frequency solutions, the equation loss magnitude vastly exceeds the data loss, so the default `γ ~ 0.5` causes the optimizer to ignore boundary conditions. Set `γ` so that both losses converge at similar rates.

**Theoretical estimate:**
$$\gamma \leq \frac{\mathcal{L}_d}{\mathcal{L}_e + \mathcal{L}_d}$$

**Algorithmic approach (Algorithm 3 — recommended):**
1. Set an initial `γ` and pre-train for `N₀` iterations.
2. Compute convergence rates: $C_d = \mathcal{L}_d^{(0)} / \mathcal{L}_d^{(min)}$ and $C_e = \mathcal{L}_e^{(0)} / \mathcal{L}_e^{(min)}$.
3. Compute the ratio $R_c = C_d / C_e$.
4. If $R_c \notin (0.1, 10)$, update: $\gamma \leftarrow \gamma \cdot R_c^\eta$ (where $\eta$ is a learning rate, e.g. 0.5).
5. Restart training from scratch with updated `γ` and repeat until criterion is met.

#### Step 4 — Build and Train the New Network

Construct the combined ansatz:

$$u_k^{(c)}(x) = u_{k-1}^{(c)}(x) + \epsilon_k \cdot u_k(x, \kappa_k)$$

where `u_k` is the new network (with sin first-layer activation and scale factor `κ_k`), and the weights/biases of all previous networks are **frozen**.

Substitute this ansatz into the original PDE and train only `u_k`. This is mathematically equivalent to solving a linearized, high-frequency version of the original equation.

#### Step 5 — Check Convergence and Repeat

Compute the new equation residue `r_{k+1}`. If it is sufficiently small, stop. Otherwise, return to Step 1 of the higher-stage procedure.

---

## Additional Settings That Improve Performance

### Optimizer Choice
- For **low-frequency** first stages: Adam + L-BFGS works well.
- For **high-frequency** higher stages: Use **Adam with Stochastic Gradient Descent (SGD)** by re-sampling collocation points every few iterations. This outperforms L-BFGS for high-frequency solutions.

### Number of Collocation Points
A sufficient number of collocation points is required to resolve high-frequency solutions:

$$N_{crit} \approx (3\pi \times \text{number of periods in domain})^d$$

where $d$ is the spatial dimension. With Adam (SGD), fewer points may suffice.

### Residual-Based Adaptive Refinement (RAR)
Continuously add collocation points in regions of high equation residue during training. This is critical when the solution has locally steep gradients.

### Gradient-Enhanced PINNs (gPINNs)
Add the gradient of the equation residue to the loss:

$$\mathcal{L}_g = \frac{1}{N_g} \sum_j |\nabla r(x_j, u(x_j))|^2$$

with weight:

$$\gamma_g \sim \frac{||r||^2}{||\nabla r||^2} \sim O(2\pi f_d)^{-2}$$

This forces the network to learn high-derivative information and reduces the power-law exponent `α` (measuring error vs. frequency), improving convergence.

---

## Expected Convergence Behavior

| Method | Error vs. Frequency (power law) | Error vs. Iterations |
|---|---|---|
| Single-stage PINN | — | $\epsilon \sim 1/n_{iters}$ (linear) |
| Multi-stage PINN | $\epsilon \sim f_d^{-1/7}$ | $\epsilon \sim \exp(-\sqrt{n_{iters}})$ |
| Multi-stage gPINN | $\epsilon \sim f_d^{-1/8}$ | Faster than PINN |

Three stages of MSNN with gPINN can typically achieve machine precision O(10⁻¹⁶) for 1D problems.

---

## Pitfalls and Edge Cases

- **Singular perturbation equations** (tiny coefficient on highest-order derivative): the magnitude estimate from Eq. (3.17) may use the wrong dominant term. Use the second-order derivative term's coefficient to estimate `ε` instead.
- **Nonlinear equations with high-order derivative products**: the linearization assumption may break down at very high frequencies. The frequency estimate from the residue is still a useful upper bound for setting `κ`.
- **Combined forward-inverse problems** (unknown PDE parameters): the prediction error may have both a high-frequency component (from the equation residue) and a low-frequency component (from the parameter inference error). Use **two networks** in the higher stage — one for each source, estimated separately.
- **Higher-dimensional problems**: convergence is slower (e.g., $\epsilon \sim \exp(-n_{iters}^{1/3})$ for 2D), but the method still outperforms single-stage training.

---

## Quick Reference: Key Formulas

| Quantity | Formula |
|---|---|
| Dominant frequency of next stage | $f_d^{(1)} \approx f_d^{(r)}$ (from FFT of residue) |
| Scale factor | $\kappa = \hat{\kappa}/\sqrt{Var}$, with $\hat{\kappa} > \pi f_d$ |
| Magnitude prefactor (order-m PDE) | $\epsilon_1 = \epsilon_{r_1} / ([2\pi f_d]^m \cdot \epsilon_\beta)$ |
| Equation weight criterion | $\gamma \leq \mathcal{L}_d / (\mathcal{L}_d + \mathcal{L}_e)$ |
| gPINN gradient loss weight | $\gamma_g \sim (2\pi f_d)^{-2}$ |
| Min collocation points | $N_c > (3\pi \cdot \text{periods})^d$ |

---

## Code Reference

The authors' implementation is available at: [https://github.com/YaoGroup/MultistageNN](https://github.com/YaoGroup/MultistageNN)
