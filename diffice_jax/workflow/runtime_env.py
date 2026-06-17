from __future__ import annotations

import json
import os
import warnings
from typing import Any


RUNTIME_ENV_KEYS = (
    "jax_compilation_cache_dir",
    "jax_enable_compilation_cache",
    "jax_explain_cache_misses",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_persistent_cache_enable_xla_caches",
    "jax_compilation_cache_include_metadata_in_key",
)
TYPO_MIN_COMPILE_TIME_KEY = "jax_persistence_cache_min_compile_time_secs"
CORRECT_MIN_COMPILE_TIME_KEY = "jax_persistent_cache_min_compile_time_secs"


def normalize_jax_platform(value: Any) -> str:
    platform = str(value).strip().lower()
    return "cuda" if platform == "gpu" else platform


def normalize_precision(value: Any) -> str:
    precision = str(value).strip().lower()
    aliases = {
        "single": "single",
        "float32": "single",
        "fp32": "single",
        "32": "single",
        "x32": "single",
        "double": "double",
        "float64": "double",
        "fp64": "double",
        "64": "double",
        "x64": "double",
    }
    try:
        return aliases[precision]
    except KeyError as exc:
        raise ValueError("runtime.precision must be 'single' or 'double'.") from exc


def runtime_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def runtime_env_updates(runtime: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    jax_platform = runtime.get("jax_platform")
    if jax_platform:
        platform = normalize_jax_platform(jax_platform)
        env["JAX_PLATFORMS"] = platform
        env["JAX_PLATFORM_NAME"] = platform

    precision = runtime.get("precision")
    if precision is not None:
        env["JAX_ENABLE_X64"] = "true" if normalize_precision(precision) == "double" else "false"

    for key in RUNTIME_ENV_KEYS:
        value = runtime.get(key)
        if value is not None:
            env[key.upper()] = runtime_env_value(value)

    typo_value = runtime.get(TYPO_MIN_COMPILE_TIME_KEY)
    if typo_value is not None:
        warnings.warn(
            f"runtime.{TYPO_MIN_COMPILE_TIME_KEY} is deprecated; "
            f"use runtime.{CORRECT_MIN_COMPILE_TIME_KEY}.",
            RuntimeWarning,
            stacklevel=2,
        )
        correct_env_key = CORRECT_MIN_COMPILE_TIME_KEY.upper()
        if correct_env_key not in env:
            env[correct_env_key] = runtime_env_value(typo_value)
    return env


def apply_runtime_env(runtime: dict[str, Any], emit: bool = False) -> dict[str, str]:
    env = runtime_env_updates(runtime)
    os.environ.update(env)
    if emit:
        print(f"JAX_CACHE_CONFIG={json.dumps(env, sort_keys=True)}", flush=True)
    return env
