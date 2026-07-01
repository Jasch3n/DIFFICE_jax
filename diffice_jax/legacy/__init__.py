"""Legacy APIs retained for older scripts and saved model bundles.

Import concrete modules such as ``diffice_jax.legacy.inverse_problem`` or
``diffice_jax.legacy.save_load`` directly. This package initializer stays
lightweight so compatibility imports do not pull in solver dependencies.
"""

__all__ = [
    "DEFAULT_GLOBAL_PARAMS",
    "DIFFICEGlobalParamsConfig",
    "DIFFICEInverseProblem",
    "build_canonical_scale",
    "ensure_canonical_scale",
    "get_default_global_params",
    "load_model",
    "resolve_gamma_c",
    "save_model",
]


def __getattr__(name):
    if name in {
        "DEFAULT_GLOBAL_PARAMS",
        "DIFFICEGlobalParamsConfig",
        "build_canonical_scale",
        "ensure_canonical_scale",
        "get_default_global_params",
        "resolve_gamma_c",
    }:
        from . import config

        return getattr(config, name)
    if name == "DIFFICEInverseProblem":
        from .inverse_problem import DIFFICEInverseProblem

        return DIFFICEInverseProblem
    if name in {"load_model", "save_model"}:
        from . import save_load

        return getattr(save_load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
