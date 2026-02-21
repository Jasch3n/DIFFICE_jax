"""
MSNN residue computation and stage-transition estimation utilities.

Given a trained PINN (or combined multi-stage PINN), this module:
  1. Evaluates the PDE equation residue at collocation points
  2. Estimates the dominant frequency (κ) from the residue via 2D FFT
  3. Estimates the magnitude prefactor (ε) from the residue RMS
  4. Estimates the equation weight (γ) from the data/equation loss ratio

Based on: Wang & Lai (2024), J. Comput. Phys. 504, 112865
"""

import jax.numpy as jnp
import numpy as np


def compute_equation_residue(predNN, gov_eqn, scale, params, x_col, idx, basal=False):
    """Evaluate the PDE residual r(x) at collocation points.

    Args:
        predNN: Forward prediction function (params, x, idx) -> solution.
        gov_eqn: Governing equation function.
        scale: Scale info for the sub-region.
        params: Network parameters (possibly combined multi-stage).
        x_col: Collocation points, shape (N, 2).
        idx: Sub-region index.
        basal: Whether this sub-region is grounded.

    Returns:
        f_eqn: Equation residual, shape (N, 2) for SSA.
    """
    net = lambda z: predNN(params, z, idx)
    f_eqn = gov_eqn(net, x_col, scale[idx], basal=basal)[0]
    return f_eqn


def estimate_kappa(residue_values, x_col, domain_range, n_hl, n_unit,
                   kappa_multiplier=1.2):
    """Estimate the scale factor κ from the dominant frequency of the residue.

    Uses 2D FFT on a gridded version of the residue to find the dominant
    frequency, then computes κ = kappa_multiplier * π * f_d / √Var,
    where Var is the Xavier variance of the first layer.

    Args:
        residue_values: Equation residual at collocation points, shape (N, n_eqn).
        x_col: Collocation point coordinates, shape (N, 2).
        domain_range: (lx0, ly0, ...) range of the normalized domain.
        n_hl: Number of hidden layers (for Xavier variance computation).
        n_unit: Number of units per hidden layer.
        kappa_multiplier: Safety factor (default 1.2).

    Returns:
        kappa: Scale factor for the correction network first layer.
        f_d: Estimated dominant frequency.
    """
    # Use the RMS of all equation components
    residue_rms_per_point = jnp.sqrt(jnp.mean(residue_values ** 2, axis=1))

    # Convert to numpy for FFT grid interpolation
    x_np = np.array(x_col)
    res_np = np.array(residue_rms_per_point)

    # Create a regular grid for FFT
    n_grid = int(np.sqrt(len(x_np)))
    n_grid = max(n_grid, 16)  # minimum grid size

    x_min, x_max = x_np[:, 0].min(), x_np[:, 0].max()
    y_min, y_max = x_np[:, 1].min(), x_np[:, 1].max()

    # Grid the scattered data (nearest-neighbor for simplicity)
    from scipy.interpolate import griddata
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    res_grid = griddata(x_np, res_np, (xi_grid, yi_grid), method='nearest')
    res_grid = np.nan_to_num(res_grid, nan=0.0)

    # 2D FFT
    fft_result = np.fft.fft2(res_grid)
    fft_magnitude = np.abs(np.fft.fftshift(fft_result))

    # Frequency axes
    dx = (x_max - x_min) / (n_grid - 1) if n_grid > 1 else 1.0
    dy = (y_max - y_min) / (n_grid - 1) if n_grid > 1 else 1.0
    freq_x = np.fft.fftshift(np.fft.fftfreq(n_grid, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(n_grid, d=dy))
    fx_grid, fy_grid = np.meshgrid(freq_x, freq_y)
    freq_mag = np.sqrt(fx_grid ** 2 + fy_grid ** 2)

    # Mask DC component
    center = n_grid // 2
    fft_magnitude[center, center] = 0.0

    # Find dominant frequency
    peak_idx = np.unravel_index(np.argmax(fft_magnitude), fft_magnitude.shape)
    f_d = float(freq_mag[peak_idx])

    # Ensure f_d is at least some minimum to avoid division by zero
    f_d = max(f_d, 1.0)

    # Xavier variance for (2, n_unit) first layer
    xavier_var = 2.0 / (2 + n_unit)

    # κ = kappa_multiplier * π * f_d / √Var
    kappa = float(kappa_multiplier * np.pi * f_d / np.sqrt(xavier_var))

    return kappa, f_d


def estimate_epsilon(residue_values, f_d, pde_order=2):
    """Estimate the magnitude prefactor ε from the residue RMS.

    ε = RMS(residue) / (2π * f_d)^m

    Args:
        residue_values: Equation residual, shape (N, n_eqn).
        f_d: Dominant frequency of the residue.
        pde_order: Order of the highest derivative in the PDE (m=2 for SSA).

    Returns:
        epsilon: Magnitude prefactor for the correction network output.
    """
    residue_rms = float(jnp.sqrt(jnp.mean(residue_values ** 2)))
    denom = (2.0 * jnp.pi * f_d) ** pde_order
    # Clamp denominator to avoid explosion
    denom = max(denom, 1e-10)
    epsilon = residue_rms / denom
    return epsilon


def estimate_gamma(loss_data, loss_eqn):
    """Estimate the equation weight γ for balanced convergence.

    γ ≤ L_d / (L_d + L_e)

    Args:
        loss_data: Current data loss value.
        loss_eqn: Current equation loss value.

    Returns:
        gamma: Recommended equation weight (upper bound).
    """
    total = loss_data + loss_eqn
    if total < 1e-30:
        return 0.5
    gamma = float(loss_data / total)
    return gamma
