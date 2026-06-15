from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from diffice_jax.core.solver import (
    DIFFICESolver,
    DataConfig,
    EquationConfig,
    LossConfig,
    ModelConfig,
    NetworkConfig,
    OptimizerConfig,
    RegionConfig,
    TrainingConfig,
    TrainingStage,
)
from diffice_jax.data.xpinns import sampling as xpinn_sampling
from diffice_jax.workflow.config import WorkflowConfig


@dataclass(frozen=True)
class WorkflowResult:
    """Result metadata from a config-driven training workflow."""

    solver: DIFFICESolver
    elapsed_seconds: float
    output_dir: Path | None


def build_solver_from_config(config: WorkflowConfig) -> DIFFICESolver:
    """Build a DIFFICESolver from a workflow config without starting training."""

    _apply_runtime_config(config.runtime)
    _apply_legacy_config(config.legacy)
    _apply_data_sampling_config(config.data)

    model = _model_config(config.model)
    public_workflow = _canonical_workflow(config.workflow)
    _validate_workflow(public_workflow, model.workflow)
    data = _data_config(config.data, model.workflow, public_workflow, config.base_dir)
    equation = EquationConfig(**dict(config.equation))
    loss = _loss_config(config.loss, config.training)
    training = _training_config(config.training, config.base_dir)
    seed = int(config.training.get("seed", config.training.get("random_seed", 0)))
    return DIFFICESolver(data=data, model=model, equation=equation, loss=loss, training=training, seed=seed)


def run_training_workflow(config: WorkflowConfig, save: bool = True) -> WorkflowResult:
    """Prepare, fit, and optionally save a config-driven training workflow."""

    solver = build_solver_from_config(config).prepare()
    start = time.perf_counter()
    solver.fit()
    elapsed = time.perf_counter() - start

    output_dir = _artifact_output_dir(config.artifacts, config.name, config.base_dir)
    if save and output_dir is not None:
        solver.save(output_dir)
    return WorkflowResult(solver=solver, elapsed_seconds=elapsed, output_dir=output_dir)


def _apply_runtime_config(runtime: dict[str, Any]) -> None:
    jax_platform = runtime.get("jax_platform")
    if jax_platform:
        os.environ.setdefault("JAX_PLATFORMS", str(jax_platform))
        os.environ.setdefault("JAX_PLATFORM_NAME", str(jax_platform))


def _apply_legacy_config(legacy: dict[str, Any]) -> None:
    xpinn_cfg = dict(legacy.get("xpinn_regression_globals", {}))
    if "interface_library" in xpinn_cfg:
        xpinn_sampling.N_INTERFACE_LIBRARY = xpinn_cfg["interface_library"]
    if "interface_collocation" in xpinn_cfg:
        xpinn_sampling.N_INTERFACE_COLLOCATION = xpinn_cfg["interface_collocation"]


def _apply_data_sampling_config(data: dict[str, Any]) -> None:
    interface_collocation = dict(data.get("interface_collocation", {}))
    if "library_size" in interface_collocation:
        xpinn_sampling.N_INTERFACE_LIBRARY = int(interface_collocation["library_size"])
    if "sample_count" in interface_collocation:
        xpinn_sampling.N_INTERFACE_COLLOCATION = int(interface_collocation["sample_count"])


def _canonical_workflow(public_workflow: str | None) -> str | None:
    if public_workflow is None:
        return None
    return public_workflow.replace("_", "-")


def _validate_workflow(public_workflow: str | None, model_workflow: str) -> None:
    if public_workflow is None:
        return
    valid = {"ice-shelf-only", "joint-inversion", "joint-inversion-regression"}
    if public_workflow not in valid:
        raise ValueError(f"Unknown workflow {public_workflow!r}. Expected one of {sorted(valid)}.")
    expected_model = "pinn" if public_workflow == "ice-shelf-only" else "xpinn"
    if model_workflow != expected_model:
        raise ValueError(
            f"Workflow {public_workflow!r} expects model.workflow={expected_model!r}, "
            f"got {model_workflow!r}."
        )


def _data_config(raw: dict[str, Any], workflow: str, public_workflow: str | None, base_dir: Path) -> DataConfig:
    cfg = dict(raw)
    cfg.pop("interface_collocation", None)
    cfg["source"] = _resolve_path(cfg["source"], base_dir)
    cfg["sampling_counts"] = _sampling_counts(cfg["sampling_counts"], workflow, public_workflow)
    if public_workflow is not None:
        cfg["regression_workflow"] = public_workflow == "joint-inversion-regression"
    return DataConfig(**cfg)


def _sampling_counts(value: Any, workflow: str, public_workflow: str | None = None) -> Any:
    if not isinstance(value, dict):
        if workflow == "pinn":
            return jnp.asarray(value, dtype="int32")
        return value

    velocity = _required_sampling_count(value, "velocity_data")
    thickness = _required_sampling_count(value, "thickness_data")
    surface = value.get("surface_data", thickness)
    collocation = _required_sampling_count(value, "collocation")
    if surface is not None and surface != thickness:
        raise ValueError("Current samplers require surface_data to match thickness_data.")

    if workflow == "pinn":
        calving_front = _required_sampling_count(value, "calving_front")
        return jnp.asarray([velocity, thickness, collocation, calving_front], dtype="int32")

    counts = [velocity, thickness, collocation]
    if public_workflow != "joint-inversion-regression":
        counts.append(_required_sampling_count(value, "calving_front"))
        counts.append(_required_sampling_count(value, "matching"))
    return counts


def _required_sampling_count(value: dict[str, Any], key: str) -> Any:
    try:
        return value[key]
    except KeyError as exc:
        raise ValueError(f"sampling_counts requires {key!r}.") from exc


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    cfg = dict(raw)
    regions = cfg.pop("regions", [])
    network = cfg.pop("network", {})
    if "per_region_networks" in cfg:
        cfg["per_region_networks"] = [_network_override(region) for region in cfg["per_region_networks"]]
    return ModelConfig(
        regions=[_region_config(region) for region in regions],
        network=NetworkConfig(**network),
        **cfg,
    )


def _region_config(raw: dict[str, Any]) -> RegionConfig:
    cfg = dict(raw)
    if "kind" in cfg:
        cfg["region_kind"] = cfg.pop("kind")
    return RegionConfig(**cfg)


def _network_override(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {"u": "net_u", "mu": "net_mu", "c": "net_c", "c0": "net_c"}
    cfg = {}
    for key, value in raw.items():
        key = aliases.get(key, key)
        if isinstance(value, (list, tuple)):
            value = {"depth": value[0], "width": value[1]}
        cfg[key] = value
    return cfg


def _loss_config(raw: dict[str, Any], training: dict[str, Any]) -> LossConfig:
    cfg = dict(raw)
    if cfg.get("name") == "joint_inversion_regression":
        cfg["name"] = "joint_inversion_regression"
    if cfg.get("name") == "joint_inversion":
        cfg["name"] = "joint_inversion"
    if "global_weights" in training:
        cfg["global_weights"] = dict(training["global_weights"])
    return LossConfig(**cfg)


def _training_config(raw: dict[str, Any], base_dir: Path) -> TrainingConfig:
    stages = []
    for raw_stage in raw.get("stages", []):
        stage = dict(raw_stage)
        if "adaptive_sampling_burnin" in stage and "adaptive_sampling_burn_in" not in stage:
            stage["adaptive_sampling_burn_in"] = stage.pop("adaptive_sampling_burnin")
        stage = {key: _normalize_scalar(value) for key, value in stage.items()}
        optimizer = dict(stage.pop("optimizer", {}))
        parameters = dict(optimizer.get("parameters", {}))
        optimizer = {key: _normalize_scalar(value) for key, value in optimizer.items()}
        optimizer["parameters"] = _optimizer_parameters(parameters, base_dir)
        stages.append(
            TrainingStage(
                optimizer=OptimizerConfig(**optimizer),
                **stage,
            )
        )
    return TrainingConfig(stages=stages)


def _optimizer_parameters(parameters: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if "use_gpinn" in parameters:
        raise ValueError("optimizer.parameters.use_gpinn is no longer supported. Use loss.use_gpinn instead.")
    for invalid_key in ("adaptive_sampling", "adaptive_sampling_burn_in", "adaptive_sampling_burnin", "adaptive_sampling_period"):
        if invalid_key in parameters:
            stage_key = "adaptive_sampling_burn_in" if invalid_key == "adaptive_sampling_burnin" else invalid_key
            raise ValueError(
                f"optimizer.parameters.{invalid_key} is no longer supported. "
                f"Use training.stages[].{stage_key} instead."
            )
    path_keys = {"checkpoint_dir", "resume_checkpoint"}
    return {
        key: _normalize_parameter_value(value, base_dir) if key in path_keys else _normalize_scalar(value)
        for key, value in parameters.items()
    }


def _normalize_parameter_value(value: Any, base_dir: Path) -> Any:
    value = _normalize_scalar(value)
    if value is None:
        return None
    return _resolve_path(value, base_dir)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == "all":
        return None
    if isinstance(value, str) and value.lower() in {"nan", ".nan"}:
        return jnp.nan
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_scalar(item) for key, item in value.items()}
    return value


def _resolve_path(value: Any, base_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _artifact_output_dir(artifacts: dict[str, Any], name: str, base_dir: Path) -> Path | None:
    output_dir = artifacts.get("output_dir")
    if output_dir is None:
        return None
    tag = artifacts.get("tag", name)
    return _resolve_path(str(output_dir).format(tag=tag, name=name), base_dir)
