from dataclasses import dataclass
import warnings

import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class DIFFICEGlobalParamsConfig:
    rho: float = 917.0
    rho_w: float = 1023.0
    g: float = 9.8
    gamma_c_default: float = 0.33

    @property
    def gd(self):
        return self.g * (1.0 - self.rho / self.rho_w)


DEFAULT_GLOBAL_PARAMS = DIFFICEGlobalParamsConfig()


def get_default_global_params():
    return DEFAULT_GLOBAL_PARAMS


def resolve_gamma_c(gamma_c=None, global_params=None, idx=None):
    gp = global_params or DEFAULT_GLOBAL_PARAMS
    if gamma_c is None:
        return gp.gamma_c_default
    if isinstance(gamma_c, (list, tuple)):
        if idx is None:
            raise ValueError("idx must be provided when gamma_c is a list/tuple")
        return gamma_c[idx]
    if hasattr(gamma_c, "shape") and len(getattr(gamma_c, "shape", [])) > 0:
        if idx is None:
            raise ValueError("idx must be provided when gamma_c is array-like")
        return gamma_c[idx]
    return gamma_c


def build_canonical_scale(dmean, drange, basal=False, gamma_c=None, global_params=None, mode="pinn"):
    gp = global_params or DEFAULT_GLOBAL_PARAMS
    gamma_c_val = resolve_gamma_c(gamma_c, gp)

    lx0, ly0, u0, v0 = drange[0:4]
    lxm, lym, um, vm = dmean[0:4]
    h0 = dmean[4]
    s0 = drange[5]
    sm = dmean[5]

    u0m = lax.max(u0, v0)
    l0m = lax.min(lx0, ly0)
    ru0 = u0 / u0m
    rv0 = v0 / u0m
    rx0 = lx0 / l0m
    ry0 = ly0 / l0m

    if mode == "xpinn":
        if basal:
            mu0 = (1.0 - gamma_c_val) * gp.rho * gp.g * s0 * (l0m / u0m)
            term0 = (1.0 - gamma_c_val) * gp.rho * gp.g * h0 * s0 / l0m / 1e3
            c0 = (gamma_c_val / (1.0 - gamma_c_val)) * (h0 * mu0) / (l0m ** 2)
            term_bd = 0.0
        else:
            mu0 = (1.0 - gp.rho / gp.rho_w) * gp.rho * gp.g * h0 * (l0m / u0m)
            term0 = gp.rho * gp.g * h0 * s0 / l0m
            c0 = jnp.nan
            term_bd = h0
    else:
        if basal:
            mu0 = gp.rho * gp.g * h0 * (l0m / u0m)
            term0 = gp.rho * gp.g * h0 ** 2 / l0m
            c0 = (h0 * mu0) / (l0m ** 2)
        else:
            mu0 = gp.rho * gp.gd * h0 * (l0m / u0m)
            term0 = gp.rho * gp.gd * h0 ** 2 / l0m
            c0 = jnp.nan
        term_bd = h0

    str0 = u0m / l0m
    du0 = str0
    dh0 = h0 / l0m

    coeff_u_x = ru0 / rx0
    coeff_u_y = ru0 / ry0
    coeff_v_x = rv0 / rx0
    coeff_v_y = rv0 / ry0
    coeff_h_x = 1.0 / rx0
    coeff_h_y = 1.0 / ry0
    coeff_s_x = 1.0 / rx0
    coeff_s_y = 1.0 / ry0

    return dict(
        lxm=lxm,
        lym=lym,
        um=um,
        vm=vm,
        lx0=lx0,
        ly0=ly0,
        u0=u0,
        v0=v0,
        h0=h0,
        s0=s0,
        sm=sm,
        u0m=u0m,
        l0m=l0m,
        ru0=ru0,
        rv0=rv0,
        rx0=rx0,
        ry0=ry0,
        du0=du0,
        dh0=dh0,
        str0=str0,
        mu0=mu0,
        c0=c0,
        term0=term0,
        term_bd=term_bd,
        gamma_c=gamma_c_val,
        coeff_u_x=coeff_u_x,
        coeff_u_y=coeff_u_y,
        coeff_v_x=coeff_v_x,
        coeff_v_y=coeff_v_y,
        coeff_h_x=coeff_h_x,
        coeff_h_y=coeff_h_y,
        coeff_s_x=coeff_s_x,
        coeff_s_y=coeff_s_y,
    )


def ensure_canonical_scale(scale, basal=False, gamma_c=None, global_params=None, mode="pinn"):
    if isinstance(scale, dict) and "u0m" in scale and "mu0" in scale:
        return scale

    if isinstance(scale, (list, tuple)) and len(scale) >= 2:
        dmean, drange = scale[0:2]
        warnings.warn(
            "Legacy scale tuple [data_mean, data_range] detected; auto-converting "
            "to canonical scale. Please migrate callers to use canonical scales.",
            DeprecationWarning,
            stacklevel=2,
        )
        return build_canonical_scale(
            dmean,
            drange,
            basal=basal,
            gamma_c=gamma_c,
            global_params=global_params,
            mode=mode,
        )

    raise TypeError("Unsupported scale format. Expected canonical dict or [data_mean, data_range].")

