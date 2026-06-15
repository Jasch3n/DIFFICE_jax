from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult


def load_mat_or_data(data: Any) -> Any:
    """Accept either an in-memory data object or a MATLAB file path."""

    if isinstance(data, (str, Path)):
        from scipy.io import loadmat

        return loadmat(str(data))
    return data


def region_kinds_to_basal_mask(region_kinds: list[str]) -> list[bool]:
    """Convert preferred region terminology to the legacy basal-mask boolean."""

    return [kind == "grounded" for kind in region_kinds]


def legacy_pinn_scale_to_subscale(data_mean, data_range, basal: bool = False) -> SubScaleResult:
    """Wrap standalone PINN mean/range arrays in the XPINN scale structure.

    The isotropic SSA equation now reads the richer ``SubScaleResult`` object.
    Older PINN preprocessing still returns positional arrays, so this adapter
    gives the solver the new contract without changing the public PINN wrapper.
    """

    data_mean = jnp.asarray(data_mean)
    data_range = jnp.asarray(data_range)
    s_mean = data_mean[5] if data_mean.shape[0] > 5 else data_mean[4]
    s_range = data_range[5] if data_range.shape[0] > 5 else data_range[4]
    mean = DataMean(data_mean[0], data_mean[1], data_mean[2], data_mean[3], data_mean[4], s_mean)
    drange = DataRange(data_range[0], data_range[1], data_range[2], data_range[3], data_range[4], s_range)

    rho = 917.0
    rho_w = 1023.0
    g = 9.8
    l0 = jnp.minimum(drange.x_range, drange.y_range)
    u0 = jnp.maximum(drange.u_range, drange.v_range)
    if basal:
        gamma_mu = 0.5
        gamma_c = 0.5
        mu0 = gamma_mu * rho * g * drange.s_range * l0 / u0
        term0 = rho * g * mean.h_mean * drange.s_range / l0
        c0 = gamma_c * term0 / u0
    else:
        gamma_mu = 0.0
        gamma_c = 0.0
        mu0 = (1.0 - rho / rho_w) * rho * g * mean.h_mean * l0 / u0
        term0 = (1.0 - rho / rho_w) * rho * g * mean.h_mean**2 / l0
        c0 = 1.0
    dynamic = DynamicScale(l0, u0, mu0, c0, term0, gamma_mu, gamma_c)
    return SubScaleResult(mean, drange, dynamic)


def to_builtin(value: Any) -> Any:
    """Convert nested config dataclasses to JSON-serializable containers."""

    if is_dataclass(value):
        return {key: to_builtin(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, jax.Array):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    return value
