from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy.io import savemat

from diffice_jax.core.adapters import (
    legacy_pinn_scale_to_subscale,
    load_mat_or_data,
    region_kinds_to_basal_mask,
    to_builtin,
)
from diffice_jax.core.contracts import FieldSchema, validate_contracts
from diffice_jax.core.loss_terms import JointInversionLossBuilder, RegressionLossBuilder
from diffice_jax.data.pinns.preprocessing import normalize_data as normalize_data_pinn
from diffice_jax.data.pinns.sampling import data_sample_create as data_sample_create_pinn
from diffice_jax.data.xpinns.preprocessing import normalize_data as normalize_data_xpinn
from diffice_jax.data.xpinns.sampling import (
    data_regression_sample_create,
    data_sample_create as data_sample_create_xpinn,
)
from diffice_jax.data.xpinns import sampling as xpinn_sampling
from diffice_jax.equation.eqn_iso import front_eqn as front_eqn_iso
from diffice_jax.equation.eqn_iso import gov_eqn as gov_eqn_iso
from diffice_jax.model.pinns.initialization import init_nets as init_pinn
from diffice_jax.model.pinns.loss import loss_iso_create as loss_iso_pinn
from diffice_jax.model.pinns.networks import solu_create as solu_pinn
from diffice_jax.model.xpinns.initialization import init_nets as init_xpinn
from diffice_jax.model.xpinns.networks import solu_create as solu_xpinn
from diffice_jax.optimizer.optimization import KfacOptimizer, adam_optimizer, lbfgs_optimizer


def _limit_rows(x, n):
    if n is None or x.shape[0] <= n:
        return x
    return x[:n]


def limit_xpinn_batch(batch, interface_points=None):
    """Limit XPINN interface rows for compact KFAC runs."""

    batch = dict(batch)
    if interface_points is not None:
        batch["md"] = [[_limit_rows(x, interface_points) for x in batch["md"][0]]]
    return batch


def attach_gpinn_interface_collocation(batch, n_regions, interface_collocation=None):
    """Attach gPINN interface collocation from the tail of each collocation set."""

    batch = dict(batch)
    n_interface = xpinn_sampling.N_INTERFACE_COLLOCATION if interface_collocation is None else interface_collocation
    gpinn_col = [
        x_col[-n_interface:] if n_interface > 0 else x_col[:0]
        for x_col in batch["col"][0][:n_regions]
    ]
    batch["gpinn_col"] = [gpinn_col]
    return batch


def replace_xpinn_collocation(batch, x_col, use_gpinn=False, n_regions=None, interface_collocation=None):
    """Replace XPINN collocation and refresh the gPINN tail when needed."""

    batch = dict(batch)
    batch["col"] = [x_col]
    if use_gpinn:
        batch = attach_gpinn_interface_collocation(batch, n_regions, interface_collocation)
    return batch


def matching_weight(step, enabled=True, start=1.0, stop=1.0, factor=1.004):
    if not enabled:
        return jnp.array(1.0)
    return jnp.minimum(start * factor ** step, stop)


def eqn_region_weights(
    step,
    idxgall,
    regions=(0, 1),
    start=1.0,
    stop=1.0,
    factor=1.0004,
):
    weights = jnp.ones(len(idxgall))
    weight = jnp.minimum(start * factor ** step, stop)
    for idx in regions:
        if idx in idxgall:
            weights = weights.at[idxgall.index(idx)].set(weight)
    return weights


def attach_xpinn_loss_weights(
    batch,
    step,
    idxgall,
    match_enabled=True,
    match_weight=1.0,
    active_regions=(0, 1),
    eqn_weight_regions=(0, 1),
):
    """Attach dynamic XPINN loss weights used by regression KFAC workflows."""

    batch = dict(batch)
    batch["match_weight"] = matching_weight(step, enabled=match_enabled, start=match_weight, stop=match_weight)
    batch["eqn_region_weights"] = eqn_region_weights(step, idxgall, regions=eqn_weight_regions)
    batch["active_regions"] = list(active_regions)
    return batch


def _batch_leaf_signature(value):
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [(str(key), _batch_leaf_signature(value[key])) for key in sorted(value, key=str)],
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_batch_leaf_signature(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_batch_leaf_signature(item) for item in value]}
    if value is None:
        return {"type": "none"}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        arr = np.asarray(value)
        shape = arr.shape
        dtype = arr.dtype
    return {
        "type": "leaf",
        "shape": [int(dim) for dim in shape],
        "dtype": str(np.dtype(dtype)),
    }


def _check_batch_leaf_signature(expected, batch, context):
    actual = _batch_leaf_signature(batch)
    if actual != expected:
        raise ValueError(
            f"KFAC batch structure changed before optim.step during {context}. "
            "This would create a new compiled variant. Expected "
            f"{json.dumps(expected, sort_keys=True)} but got {json.dumps(actual, sort_keys=True)}."
        )
    return actual


def _json_ready(value):
    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, dict):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.asarray(value)
        if arr.shape == ():
            return _json_ready(arr.item())
        return {"shape": [int(dim) for dim in arr.shape], "dtype": str(arr.dtype)}
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return value


def _jax_cache_env_signature():
    keys = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_EXPLAIN_CACHE_MISSES",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY",
        "JAX_ENABLE_X64",
    )
    env = {}
    for key in keys:
        if key in os.environ:
            env[key] = "<set>" if key.endswith("_DIR") else os.environ[key]
    return env


@dataclass(frozen=True)
class DataConfig:
    """Input-data and sampling configuration for a solver run."""

    source: Any
    sampling_counts: Any
    collocation_library_size: int | Literal["full"] | None = None
    validation: Any | None = None
    regression_workflow: bool = False
    grounded_only_interface_mu_ct: bool = False
    interface_mu_source: str = "floating"
    legacy_matlab_keys: bool = True


@dataclass(frozen=True)
class RegionConfig:
    """One PINN/XPINN region and its ice-domain kind."""

    region_kind: Literal["floating", "grounded"]
    index: int
    legacy_index: int | None = None


@dataclass(frozen=True)
class NetworkConfig:
    """Default neural-network architecture shared by one or more regions."""

    depth: int = 4
    width: int = 25
    activation: int = 0
    first_layer_scale: float = 1.0
    embedding: bool = False
    embed_n: int = 10
    embed_std: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """Model-family and architecture configuration."""

    workflow: Literal["pinn", "xpinn"]
    regions: list[RegionConfig] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    per_region_networks: list[dict[str, Any]] | None = None
    per_region_embeddings: list[dict[str, Any]] | None = None
    anisotropic: bool = False


@dataclass(frozen=True)
class EquationConfig:
    """Equation and boundary-condition selection."""

    name: Literal["ssa_iso"] = "ssa_iso"
    boundary_condition: Literal["calving_front"] | None = "calving_front"
    vanilla_floating_ssa: bool = True


@dataclass(frozen=True)
class LossConfig:
    """Loss-term selection and high-level term weights."""

    name: Literal["iso", "joint_inversion", "joint_inversion_regression", "regression"] = "joint_inversion"
    weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    matching: bool = True
    calving_front: bool = True
    match_weight: float = 1.0
    match_component_weights: Any | None = None
    use_gpinn: bool = False
    gpinn_weight: float = 0.0
    mu_grad_weight: float = 0.0
    active_regions: Any | None = None
    global_weights: dict[str, Any] | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer selection plus optimizer-specific scalar controls."""

    name: Literal["adam", "lbfgs", "kfac"] = "adam"
    learning_rate: float = 1e-3
    damping: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingStage:
    """One ordered optimization stage in a training run."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    iterations: int = 1000
    adaptive_sampling: bool = False
    adaptive_sampling_burn_in: int = 1000
    adaptive_sampling_period: int = 50
    checkpoint_interval: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Full ordered training schedule."""

    stages: list[TrainingStage] = field(default_factory=lambda: [TrainingStage()])


@dataclass
class PreparedState:
    """Mutable runtime state produced by ``DIFFICESolver.prepare``."""

    raw_data: Any
    normalized_data: Any
    sub_region_indices: list[int] | None
    position_data: Any | None
    crop_indices: Any | None
    basal_mask: list[bool]
    scales: Any
    params: Any
    solution: Any
    dataf: Any
    lossf: Any
    initial_loss: Any


class DIFFICESolver:
    """High-level orchestration object for DIFFICE_jax workflows."""

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        equation: EquationConfig = EquationConfig(),
        loss: LossConfig = LossConfig(),
        training: TrainingConfig = TrainingConfig(),
        seed: int = 0,
    ):
        self.data_config = data
        self.model_config = model
        self.equation_config = equation
        self.loss_config = loss
        self.training_config = training
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.state: PreparedState | None = None
        self.loss_history: list[Any] = []
        self.predictions: dict[str, Any] | None = None

    def prepare(self):
        """Normalize data, initialize networks, create samplers, and bind losses."""

        if self.model_config.workflow == "xpinn":
            self.state = self._prepare_xpinn()
        elif self.model_config.workflow == "pinn":
            self.state = self._prepare_pinn()
        else:
            raise ValueError(f"Unknown workflow: {self.model_config.workflow}")
        return self

    def fit(self):
        """Run configured optimization stages in order."""

        state = self._require_prepared()
        params = state.params
        key = self.key
        for stage in self.training_config.stages:
            name = stage.optimizer.name
            if name == "adam":
                key, stage_key = random.split(key)
                params, history = adam_optimizer(
                    stage_key,
                    state.lossf,
                    params,
                    state.dataf,
                    stage.iterations,
                    lr=stage.optimizer.learning_rate,
                    aniso=self.model_config.anisotropic,
                    eval_f=self._adaptive_eval(params),
                    adaptive=stage.adaptive_sampling,
                    adapt_period=stage.adaptive_sampling_period,
                    adapt_burnin=stage.adaptive_sampling_burn_in,
                )
            elif name == "lbfgs":
                key, stage_key = random.split(key)
                data = state.dataf(stage_key)
                params, history = lbfgs_optimizer(
                    state.lossf,
                    params,
                    data,
                    stage.iterations,
                    basal=any(state.basal_mask),
                )
            elif name == "kfac":
                params, history = self._fit_kfac_stage(params, state.dataf, state.lossf, stage, key)
                key = getattr(self, "kfac_state", {}).get("key", key)
            else:
                raise ValueError(f"Unknown optimizer: {name}")
            self.loss_history.extend(history)
        self.key = key
        state.params = params
        return self

    def load_params(self, path):
        """Load saved parameters into a prepared solver."""

        state = self._require_prepared()
        with open(path, "rb") as f:
            params = pickle.load(f)
        if isinstance(params, dict) and "params" in params:
            params = params["params"]
        state.params = params
        return self

    def predict(self):
        """Return inferred fields using the prepared/trained parameters."""

        state = self._require_prepared()
        if self.model_config.workflow == "xpinn":
            self.predictions = {
                "workflow": "xpinn",
                "regions": self._predict_xpinn_fields(state),
            }
        else:
            self.predictions = {
                "workflow": "pinn",
                "regions": [self._predict_pinn_fields(state)],
            }
        return self.predictions

    def predict_equation_diagnostics(self, points: str = "velocity"):
        """Evaluate equation residuals and terms per region.

        ``points`` selects normalized evaluation coordinates from the prepared
        data. Use ``"velocity"`` for the same points as inferred-field output
        and ``"collocation"`` for XPINN collocation libraries.
        """

        state = self._require_prepared()
        if self.model_config.workflow == "xpinn":
            diagnostics = {
                "workflow": "xpinn",
                "points": points,
                "regions": self._xpinn_equation_diagnostics(state, points),
            }
        else:
            diagnostics = {
                "workflow": "pinn",
                "points": points,
                "regions": [self._pinn_equation_diagnostics(state, points)],
            }
        return diagnostics

    def _predict_xpinn_fields(self, state):
        regions = []
        for pos, idx in enumerate(state.sub_region_indices):
            data = state.normalized_data[idx]
            X = data[0][0]
            Xh = data[0][1]
            Xs = data[0][3]
            raw = data[4][3]
            idxval = jnp.asarray(data[4][4][0], dtype=jnp.int32).reshape(-1)
            idxval_h = jnp.asarray(data[4][4][1], dtype=jnp.int32).reshape(-1)
            idxval_s = jnp.asarray(data[4][4][2], dtype=jnp.int32).reshape(-1)
            scale = state.scales[pos]
            dmean = scale.data_mean
            drange = scale.data_range
            dynamic = scale.dynamic_scale
            output = state.solution[0](state.params, X, idx)
            output_h = state.solution[0](state.params, Xh, idx)
            output_s = state.solution[0](state.params, Xs, idx)
            basal = bool(state.basal_mask[pos])
            C = output[:, 5:6] * dynamic.c0 if basal and output.shape[1] > 5 else jnp.full_like(output[:, 4:5], jnp.nan)
            region = {
                "index": idx,
                "kind": "grounded" if basal else "floating",
                "x": jnp.asarray(raw[0]).reshape(-1, 1)[idxval],
                "y": jnp.asarray(raw[1]).reshape(-1, 1)[idxval],
                "x_h": jnp.asarray(raw[4]).reshape(-1, 1)[idxval_h],
                "y_h": jnp.asarray(raw[5]).reshape(-1, 1)[idxval_h],
                "x_s": jnp.asarray(state.raw_data["xd_s"][0, idx]).reshape(-1, 1)[idxval_s],
                "y_s": jnp.asarray(state.raw_data["yd_s"][0, idx]).reshape(-1, 1)[idxval_s],
                "u": output[:, 0:1] * drange.u_range + dmean.u_mean,
                "v": output[:, 1:2] * drange.v_range + dmean.v_mean,
                "h": output[:, 2:3] * dmean.h_mean,
                "h_thickness": output_h[:, 2:3] * dmean.h_mean,
                "s": output[:, 3:4] * drange.s_range + dmean.s_mean,
                "s_thickness": output_h[:, 3:4] * drange.s_range + dmean.s_mean,
                # Surface elevation's own prediction, evaluated at its own
                # native xd_s/yd_s points (independent of the thickness grid
                # — see data_sources/surface.py in joint_xpinn_data). Unlike
                # s_thickness (evaluated at the thickness points, only
                # meaningful if xd_s happens to coincide with xd_h), this is
                # always directly comparable to the observed `sd` truth.
                "s_surface": output_s[:, 3:4] * drange.s_range + dmean.s_mean,
                "mu": output[:, 4:5] * dynamic.mu0,
                "C": C,
            }
            if len(raw) > 12:
                region["mu_true"] = jnp.asarray(raw[12]).reshape(-1, 1)[idxval]
            if len(raw) > 13:
                region["C_true"] = jnp.asarray(raw[13]).reshape(-1, 1)[idxval]
            regions.append(region)
        return regions

    def _predict_pinn_fields(self, state):
        basal = any(state.basal_mask)
        data = state.normalized_data
        raw = data[4][3]
        idxval = jnp.asarray(data[4][4][0], dtype=jnp.int32).reshape(-1)
        idxval_h = jnp.asarray(data[4][4][1], dtype=jnp.int32).reshape(-1)
        scale = state.scales[0]
        dmean = scale.data_mean
        drange = scale.data_range
        dynamic = scale.dynamic_scale
        X = data[0][0]
        Xh = data[0][1]
        output = state.solution(state.params, X)
        output_h = state.solution(state.params, Xh)
        mu_col = 4 if basal else 3
        C = output[:, 5:6] * dynamic.c0 if basal and output.shape[1] > 5 else None
        region = {
            "index": 0,
            "kind": "grounded" if basal else "floating",
            "x": jnp.asarray(raw[0]).reshape(-1, 1)[idxval],
            "y": jnp.asarray(raw[1]).reshape(-1, 1)[idxval],
            "x_h": jnp.asarray(raw[4]).reshape(-1, 1)[idxval_h],
            "y_h": jnp.asarray(raw[5]).reshape(-1, 1)[idxval_h],
            "u": output[:, 0:1] * drange.u_range + dmean.u_mean,
            "v": output[:, 1:2] * drange.v_range + dmean.v_mean,
            "h": output[:, 2:3] * dmean.h_mean,
            "h_thickness": output_h[:, 2:3] * dmean.h_mean,
            "mu": output[:, mu_col:mu_col + 1] * dynamic.mu0,
        }
        if basal:
            region["s"] = output[:, 3:4] * dmean.h_mean
            region["s_thickness"] = output_h[:, 3:4] * dmean.h_mean
            region["C"] = C
        else:
            region["s"] = jnp.full_like(output[:, 2:3], jnp.nan)
            region["s_thickness"] = jnp.full_like(output_h[:, 2:3], jnp.nan)
            region["C"] = jnp.full_like(output[:, mu_col:mu_col + 1], jnp.nan)
        return region

    def _xpinn_equation_diagnostics(self, state, points):
        regions = []
        for pos, idx in enumerate(state.sub_region_indices):
            data = state.normalized_data[idx]
            scale = state.scales[pos]
            X = self._xpinn_eval_points(data, scale, points)
            raw_x, raw_y = self._denormalize_xy(X, scale)
            basal = bool(state.basal_mask[pos])
            net = lambda z: state.solution[0](state.params, z, idx)
            residual, terms = jax.vmap(lambda z: gov_eqn_iso(net, z, scale, basal=basal), in_axes=(0,))(X)
            regions.append(self._equation_region(idx, "grounded" if basal else "floating", raw_x, raw_y, residual, terms))
        return regions

    def _pinn_equation_diagnostics(self, state, points):
        basal = any(state.basal_mask)
        scale = state.scales[0]
        X = self._pinn_eval_points(state.normalized_data, scale, points)
        raw_x, raw_y = self._denormalize_xy(X, scale)
        net = lambda z: state.solution(state.params, z)
        residual, terms = jax.vmap(lambda z: gov_eqn_iso(net, z, scale, basal=basal), in_axes=(0,))(X)
        return self._equation_region(0, "grounded" if basal else "floating", raw_x, raw_y, residual, terms)

    @staticmethod
    def _xpinn_eval_points(data, scale, points):
        if points == "velocity":
            return data[0][0]
        if points == "thickness":
            return data[0][1]
        if points == "collocation":
            return data[0][2]
        raise ValueError("points must be 'velocity', 'thickness', or 'collocation'.")

    @staticmethod
    def _pinn_eval_points(data, scale, points):
        if points == "velocity":
            return data[0][0]
        if points == "thickness":
            return data[0][1]
        raise ValueError("points must be 'velocity' or 'thickness' for PINN diagnostics.")

    @staticmethod
    def _denormalize_xy(X, scale):
        return (
            X[:, 0:1] * scale.data_range.x_range + scale.data_mean.x_mean,
            X[:, 1:2] * scale.data_range.y_range + scale.data_mean.y_mean,
        )

    @staticmethod
    def _equation_region(idx, kind, x, y, residual, terms):
        names = (
            ("viscous_x_x", "viscous_x_y", "driving_x", "viscous_y_y", "viscous_y_x", "driving_y", "strain_rate",
             "basal_x", "basal_y")
            if terms.shape[1] > 7 else
            ("viscous_x_x", "viscous_x_y", "driving_x", "viscous_y_y", "viscous_y_x", "driving_y", "strain_rate")
        )
        term_dict = {name: terms[:, pos:pos + 1] for pos, name in enumerate(names)}
        return {
            "index": idx,
            "kind": kind,
            "x": x,
            "y": y,
            "residual_x": residual[:, 0:1],
            "residual_y": residual[:, 1:2],
            "residual_magnitude": jnp.sqrt(jnp.sum(jnp.square(residual), axis=1, keepdims=True)),
            "terms": term_dict,
            "term_magnitudes": {name: jnp.abs(value) for name, value in term_dict.items()},
            "raw_terms": terms,
        }

    def save(self, output_dir):
        """Write parameters, configs, loss history, and predictions to disk."""

        state = self._require_prepared()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "params.pkl", "wb") as f:
            pickle.dump(state.params, f, pickle.HIGHEST_PROTOCOL)
        with open(output_dir / "loss_history.pkl", "wb") as f:
            pickle.dump(self.loss_history, f, pickle.HIGHEST_PROTOCOL)
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data": to_builtin(self.data_config),
                    "model": to_builtin(self.model_config),
                    "equation": to_builtin(self.equation_config),
                    "loss": to_builtin(self.loss_config),
                    "training": to_builtin(self.training_config),
                    "seed": self.seed,
                },
                f,
                indent=2,
            )
        predictions = self.predict()
        with open(output_dir / "predictions.pkl", "wb") as f:
            pickle.dump(jax.device_get(predictions), f, pickle.HIGHEST_PROTOCOL)
        try:
            savemat(output_dir / "predictions.mat", predictions)
        except TypeError:
            pass
        return output_dir

    def _prepare_xpinn(self):
        """Prepare an XPINN workflow while preserving legacy data structures."""

        raw_data = load_mat_or_data(self.data_config.source)
        regions = self._xpinn_regions(raw_data)
        basal_mask = region_kinds_to_basal_mask([region.region_kind for region in regions])
        self._validate_schemas(basal_mask)
        normalized_data, sub_region_indices, position_data, crop_indices = normalize_data_xpinn(
            raw_data,
            basal_mask=basal_mask,
            use_regression=self.data_config.regression_workflow,
            grounded_only_interface_mu_ct=self.data_config.grounded_only_interface_mu_ct,
            interface_mu_source=self.data_config.interface_mu_source,
        )
        scales = [normalized_data[idx][4][-1] for idx in sub_region_indices]
        init_key, key = random.split(self.key)
        data_key, key = random.split(key)
        self.key = key
        net = self.model_config.network
        params = init_xpinn(
            init_key,
            net.depth,
            net.width,
            n_sub=len(sub_region_indices),
            aniso=self.model_config.anisotropic,
            basal_mask=basal_mask,
            embedding_config=self.model_config.per_region_embeddings,
            network_config=self.model_config.per_region_networks,
        )
        solution = solu_xpinn(
            scales,
            scl=net.first_layer_scale,
            act_s=net.activation,
            basal_mask=basal_mask,
        )
        if self.data_config.regression_workflow:
            dataf = data_regression_sample_create(
                normalized_data,
                sub_region_indices,
                self.data_config.sampling_counts,
                basal_mask=basal_mask,
                grounded_only_interface_mu_ct=self.data_config.grounded_only_interface_mu_ct,
            )
        else:
            dataf = data_sample_create_xpinn(
                normalized_data,
                sub_region_indices,
                self.data_config.sampling_counts,
                basal_mask=basal_mask,
                use_regression=False,
            )
        lossf = self._xpinn_loss(solution, sub_region_indices, basal_mask, scales)
        first_batch = dataf(data_key)
        initial_loss = lossf(params, first_batch)[0]
        lossf.lref = initial_loss
        return PreparedState(
            raw_data,
            normalized_data,
            sub_region_indices,
            position_data,
            crop_indices,
            basal_mask,
            scales,
            params,
            solution,
            dataf,
            lossf,
            initial_loss,
        )

    def _prepare_pinn(self):
        """Prepare a standalone PINN workflow with adapted scale metadata."""

        raw_data = load_mat_or_data(self.data_config.source)
        basal = bool(self.model_config.regions and self.model_config.regions[0].region_kind == "grounded")
        basal_mask = [basal]
        self._validate_schemas(basal_mask)
        normalized_data = normalize_data_pinn(raw_data, basal=basal)
        loss_scale = legacy_pinn_scale_to_subscale(normalized_data[4][0], normalized_data[4][1], basal=basal)
        key, init_key, data_key = random.split(self.key, 3)
        self.key = key
        net = self.model_config.network
        params = init_pinn(
            init_key,
            net.depth,
            net.width,
            aniso=self.model_config.anisotropic,
            basal=basal,
            embedding=net.embedding,
            embed_n=net.embed_n,
            embed_std=net.embed_std,
        )
        solution = solu_pinn(
            scl=net.first_layer_scale,
            act_s=net.activation,
            basal=basal,
            embedding=net.embedding,
        )
        dataf = data_sample_create_pinn(
            normalized_data,
            self.data_config.sampling_counts,
            basal=basal,
            collocation_library_size=self.data_config.collocation_library_size,
        )
        lossf = loss_iso_pinn(solution, (gov_eqn_iso, front_eqn_iso), loss_scale, self.loss_config.weights, basal=basal)
        first_batch = dataf(data_key)
        initial_loss = lossf(params, first_batch)[0]
        lossf.lref = initial_loss
        return PreparedState(
            raw_data,
            normalized_data,
            None,
            None,
            None,
            basal_mask,
            [loss_scale],
            params,
            solution,
            dataf,
            lossf,
            initial_loss,
        )

    def _xpinn_loss(self, solution, sub_region_indices, basal_mask, scales):
        """Create the XPINN loss selected by ``LossConfig``."""

        if self.loss_config.name == "joint_inversion":
            builder = JointInversionLossBuilder(self.loss_config)
            return builder.create(
                solution,
                sub_region_indices,
                basal_mask,
                gov_eqn_iso,
                front_eqn_iso,
                scales,
            )
        if self.loss_config.name in ("joint_inversion_regression", "regression"):
            builder = RegressionLossBuilder(self.loss_config, self.data_config)
            return builder.create(
                solution,
                sub_region_indices,
                basal_mask,
                gov_eqn_iso,
                front_eqn_iso,
                scales,
            )
        raise ValueError(f"Unsupported XPINN loss: {self.loss_config.name}")

    def _fit_kfac_stage(self, params, dataf, lossf, stage, key):
        """Run one KFAC stage with loss registration for curvature estimates."""

        from kfac_jax import loss_functions as kfac_loss_functions

        controls = dict(stage.optimizer.parameters)
        preset = controls.pop("preset", None)
        log_rate = controls.pop("log_rate", 100)
        legacy_kfac_eval = controls.pop("legacy_kfac_eval", False)
        interface_points = controls.pop("interface_points", None)
        if "calving_front_points" in controls:
            raise ValueError(
                "optimizer.parameters.calving_front_points is no longer supported. "
                "Use data.sampling_counts.calving_front to control calving-front samples."
            )
        interface_collocation = controls.pop("interface_collocation", None)
        if "continue_chunk" in controls:
            raise ValueError(
                "optimizer.parameters.continue_chunk is no longer supported. "
                "Set training.stages[].iterations to the absolute target iteration instead."
            )
        max_iterations = controls.pop("max_iterations", None)
        checkpoint_dir = controls.pop("checkpoint_dir", None)
        resume_checkpoint = controls.pop("resume_checkpoint", None)
        target_c_rel_mae = controls.pop("target_c_rel_mae", None)
        if "use_gpinn" in controls:
            raise ValueError("optimizer.parameters.use_gpinn is no longer supported. Use loss.use_gpinn instead.")
        use_gpinn = self._effective_gpinn_enabled()
        for invalid_key in ("adaptive_sampling", "adaptive_sampling_burn_in", "adaptive_sampling_burnin", "adaptive_sampling_period"):
            if invalid_key in controls:
                stage_key = "adaptive_sampling_burn_in" if invalid_key == "adaptive_sampling_burnin" else invalid_key
                raise ValueError(
                    f"optimizer.parameters.{invalid_key} is no longer supported. "
                    f"Use training.stages[].{stage_key} instead."
                )
        adaptive_sampling = stage.adaptive_sampling
        adaptive_sampling_burn_in = stage.adaptive_sampling_burn_in
        adaptive_sampling_period = stage.adaptive_sampling_period
        checkpoint_callback = controls.pop("checkpoint_callback", None)
        eval_f = controls.pop("eval_f", self._adaptive_eval(params))
        n_regions = controls.pop(
            "n_regions",
            1 if self.state is None or self.state.sub_region_indices is None else len(self.state.sub_region_indices),
        )
        idxgall = controls.pop(
            "idxgall",
            list(range(n_regions)) if self.state is None or self.state.sub_region_indices is None else list(self.state.sub_region_indices),
        )
        match_enabled = controls.pop("match_enabled", self.loss_config.matching)
        active_regions = controls.pop("active_regions", idxgall)
        eqn_weight_regions = controls.pop("eqn_weight_regions", idxgall)
        global_weights = {} if self.loss_config.global_weights is None else dict(self.loss_config.global_weights)
        match_weight = global_weights.get("matching", global_weights.get("match", self.loss_config.match_weight))

        config = dict(
            learning_rate=stage.optimizer.learning_rate,
            momentum=0.9,
            damping=stage.optimizer.damping if stage.optimizer.damping is not None else jnp.nan,
            norm_constraint=1e-3,
            initial_damping=1e-3,
            min_damping=1e-6,
            curvature_block_type="naive_full",
            damping_adaptation_decay=0.998,
            curvature_ema=0.95,
            inverse_update_period=1,
            num_burnin_steps=0,
            always_use_exact_qmodel_for_damping_adjustment=True,
            include_norms_in_stats=True,
        )
        if preset == "xpinn_joint_inversion_reference":
            config.update(
                learning_rate=None,
                momentum=None,
                damping=jnp.nan,
                norm_constraint=1e-8,
                initial_damping=1,
                min_damping=1e-6,
                curvature_block_type="naive_full",
                damping_adaptation_decay=0.997,
                curvature_ema=0.997,
                inverse_update_period=10,
                num_burnin_steps=0,
                always_use_exact_qmodel_for_damping_adjustment=True,
                include_norms_in_stats=True,
            )
        config.update(controls)

        params = dict(params) if hasattr(params, "keys") else params

        def use_kfac_eval():
            return hasattr(lossf, "kfac_eval") and not legacy_kfac_eval

        def objective_value(current_params, batch):
            if use_kfac_eval():
                return lossf.kfac_eval(current_params, batch)[0]
            if hasattr(lossf, "kfac_residuals"):
                residuals = lossf.kfac_residuals(current_params, batch)
                return jnp.sum(jnp.square(residuals)) / lossf.lref
            return lossf(current_params, batch)[0]

        def kfac_lossf(current_params, batch):
            if use_kfac_eval():
                loss_n, loss_info, _, raw_residuals = lossf.kfac_eval(current_params, batch)
                residuals = raw_residuals / jnp.sqrt(lossf.lref)
            elif hasattr(lossf, "kfac_residuals"):
                raw_residuals = lossf.kfac_residuals(current_params, batch)
                loss_out = lossf(current_params, batch)
                loss_info = loss_out[1]
                residuals = raw_residuals / jnp.sqrt(lossf.lref)
                loss_n = jnp.sum(jnp.square(residuals))
            else:
                loss_scalar, loss_info = lossf(current_params, batch)
                loss_n = loss_scalar
                residuals = jnp.sqrt(jnp.maximum(loss_n, 0.0)).reshape(1, 1)
            # Match the XPINN regression KFAC path: register one weighted
            # residual vector so KFAC sees a single least-squares objective.
            kfac_loss_functions.register_squared_error_loss(
                residuals,
                targets=jnp.zeros_like(residuals),
            )
            return loss_n, loss_info

        def sampled_batch(batch_key, step, **sample_kwargs):
            batch = dataf(batch_key, **sample_kwargs)
            batch = limit_xpinn_batch(batch, interface_points)
            if use_gpinn:
                batch = attach_gpinn_interface_collocation(batch, n_regions, interface_collocation)
            return attach_xpinn_loss_weights(
                batch,
                step,
                idxgall,
                match_enabled=match_enabled,
                match_weight=match_weight,
                active_regions=active_regions,
                eqn_weight_regions=eqn_weight_regions,
            )

        def format_seconds(seconds):
            seconds = float(seconds)
            return "nan" if not np.isfinite(seconds) else f"{seconds:.1f}s"

        def format_scientific(value):
            return f"{float(np.asarray(value)):.3e}"

        def print_rad_diagnostics(step, diagnostics):
            if diagnostics is None:
                return
            for item in diagnostics:
                print(
                    f"KFAC step {step} | rad_precision | "
                    f"region={int(item['region'])} | "
                    f"eps={format_scientific(item['eps'])} | "
                    f"res_min={format_scientific(item['res_min'])} | "
                    f"p(res_min)={format_scientific(item['prob_at_res_min'])} | "
                    f"res_max={format_scientific(item['res_max'])} | "
                    f"p(res_max)={format_scientific(item['prob_at_res_max'])} | "
                    f"res_min/(eps*res_max)={format_scientific(item['res_min_roundoff_ratio'])} | "
                    f"prob_min={format_scientific(item['prob_min'])} | "
                    f"prob_max={format_scientific(item['prob_max'])} | "
                    f"(prob_max-prob_min)/(eps*prob_max)={format_scientific(item['prob_span_roundoff_ratio'])}",
                    flush=True,
                )

        key, init_key = random.split(key)
        init_data = sampled_batch(init_key, 0)
        batch_signature = _batch_leaf_signature(init_data)
        print(
            "KFAC_COMPILE_SIGNATURE="
            + json.dumps(
                self._kfac_compile_signature(
                    batch_signature,
                    config,
                    stage,
                    use_gpinn=use_gpinn,
                    active_regions=active_regions,
                    eqn_weight_regions=eqn_weight_regions,
                    idxgall=idxgall,
                    n_regions=n_regions,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        optim = KfacOptimizer(loss_fn=kfac_lossf, **config).get_optimizer()
        opt_state = optim.init(params, init_key, init_data)
        initial_objective = float(objective_value(params, init_data))
        history = [(0, initial_objective)]
        damping = config["initial_damping"]
        damping_decay = config["damping_adaptation_decay"]
        damping_min = config["min_damping"]
        use_step_damping = bool(jnp.isnan(jnp.asarray(config["damping"])))
        x_col_mem = None
        adapted = False
        final_step = 0
        checkpoint_path = None
        interface_point_counts = np.asarray([x.shape[0] for x in init_data.get("md", [[]])[0]], dtype=int)

        if resume_checkpoint is not None:
            resume_checkpoint = Path(resume_checkpoint)
            with open(resume_checkpoint, "rb") as f:
                checkpoint = pickle.load(f)
            params = checkpoint["params"]
            opt_state = checkpoint.get("opt_state", opt_state)
            damping = float(np.asarray(checkpoint.get("damping", damping)))
            key = jnp.asarray(checkpoint.get("key", key))
            final_step = int(checkpoint["step"])
            checkpoint_history = checkpoint.get("loss_history", None)
            if checkpoint_history is not None:
                history = [tuple(x) for x in np.asarray(checkpoint_history, dtype=float)]
                initial_objective = float(history[0][1]) if history else initial_objective
            x_col_mem = checkpoint.get("x_col_mem", None)
            adapted = bool(checkpoint.get("adapted", x_col_mem is not None))
            checkpoint_path = resume_checkpoint
            print(f"KFAC resumed from {resume_checkpoint} at step {final_step}", flush=True)

        start_step = final_step
        start = time.perf_counter()
        timed_start = None
        timed_start_step = None
        while True:
            segment_end = stage.iterations
            if max_iterations is not None:
                segment_end = min(segment_end, max_iterations)
            for step in range(final_step, segment_end):
                key, step_key, data_key = random.split(key, 3)
                run_rad = (
                    adaptive_sampling
                    and (step + 1) % adaptive_sampling_period == 0
                    and (step + 1) > adaptive_sampling_burn_in
                )
                if run_rad:
                    batch_context = f"RAD step {step + 1}"
                    rad_start = time.perf_counter()
                    print(
                        f"KFAC step {step + 1} | adaptive_sampling=start | "
                        f"burn_in={adaptive_sampling_burn_in} | period={adaptive_sampling_period}",
                        flush=True,
                    )
                    batch = sampled_batch(
                        data_key,
                        step,
                        eval_adaptive=True,
                        eval_f=None if eval_f is None else lambda x, idx, basal: eval_f(params, x, idx, basal),
                    )
                    print_rad_diagnostics(step + 1, batch.pop("rad_diagnostics", None))
                    x_col_mem = batch["col"][0]
                    adapted = True
                    print(
                        f"KFAC step {step + 1} | adaptive_sampling=done | "
                        f"elapsed={format_seconds(time.perf_counter() - rad_start)}",
                        flush=True,
                    )
                elif adaptive_sampling and adapted:
                    batch_context = f"reused adaptive step {step + 1}"
                    batch = sampled_batch(data_key, step, eval_adaptive=False)
                    batch = replace_xpinn_collocation(
                        batch,
                        x_col_mem,
                        use_gpinn=use_gpinn,
                        n_regions=n_regions,
                        interface_collocation=interface_collocation,
                    )
                else:
                    batch_context = f"sampled step {step + 1}"
                    batch = sampled_batch(data_key, step)
                _check_batch_leaf_signature(batch_signature, batch, batch_context)
                step_kwargs = dict(batch=batch, global_step_int=step)
                if use_step_damping:
                    step_kwargs["damping"] = damping
                params, opt_state, stats = optim.step(params, opt_state, step_key, **step_kwargs)
                if timed_start is None:
                    timed_start = time.perf_counter()
                    timed_start_step = step + 1
                if use_step_damping and damping > damping_min:
                    damping *= damping_decay
                if step < 2 or (step + 1) % log_rate == 0 or step + 1 == segment_end:
                    current_objective = float(objective_value(params, batch))
                    history.append((step + 1, current_objective))
                    loss_info = stats["aux"]
                    total_elapsed = time.perf_counter() - start
                    timed_elapsed = 0.0 if timed_start is None else time.perf_counter() - timed_start
                    timed_iterations = 0 if timed_start_step is None else step + 1 - timed_start_step
                    seconds_per_iter = timed_elapsed / timed_iterations if timed_iterations else np.nan
                    seconds_per_iter_label = "warming_up" if timed_iterations == 0 else format_seconds(seconds_per_iter)
                    print(
                        f"KFAC step {step + 1} | "
                        f"objective={current_objective:.4e} | "
                        f"scalar_loss={float(loss_info[0]):.4e} | "
                        f"damping={float(stats.get('damping', damping)):.2e} | "
                        f"elapsed={format_seconds(total_elapsed)} | "
                        f"timed_elapsed={format_seconds(timed_elapsed)} | "
                        f"timed_iterations={timed_iterations} | "
                        f"seconds_per_iter={seconds_per_iter_label}",
                        flush=True,
                    )

            final_step = segment_end
            state = dict(
                step=final_step,
                params=params,
                opt_state=opt_state,
                damping=damping,
                key=key,
                history=history,
                x_col_mem=x_col_mem,
                adapted=adapted,
                checkpoint_dir=checkpoint_dir,
                checkpoint_path=checkpoint_path,
                interface_point_counts=interface_point_counts,
                initial_objective=initial_objective,
                start_step=start_step,
                elapsed=time.perf_counter() - start,
                timed_elapsed=0.0 if timed_start is None else time.perf_counter() - timed_start,
                timed_iterations=0 if timed_start_step is None else final_step - timed_start_step,
            )
            if checkpoint_callback is not None:
                callback_result = checkpoint_callback(state)
                if callback_result is not None:
                    checkpoint_path = callback_result.get("checkpoint_path", checkpoint_path)
                    if target_c_rel_mae is not None and callback_result.get("c_rel_mae", jnp.inf) < target_c_rel_mae:
                        break
            break

        rate_timed_elapsed = 0.0 if timed_start is None else time.perf_counter() - timed_start
        rate_timed_iterations = 0 if timed_start_step is None else final_step - timed_start_step
        final_batch = sampled_batch(key, final_step)
        final_objective = float(objective_value(params, final_batch))
        history.append((final_step, final_objective))
        elapsed = time.perf_counter() - start
        self.kfac_state = dict(
            params=params,
            opt_state=opt_state,
            damping=damping,
            key=key,
            history=history,
            initial_objective=initial_objective,
            final_objective=final_objective,
            start_step=start_step,
            final_step=final_step,
            elapsed=elapsed,
            timed_elapsed=rate_timed_elapsed,
            timed_iterations=rate_timed_iterations,
            seconds_per_iter=rate_timed_elapsed / rate_timed_iterations if rate_timed_iterations else np.nan,
            checkpoint_path=checkpoint_path,
            x_col_mem=x_col_mem,
            adapted=adapted,
            interface_point_counts=interface_point_counts,
        )
        return params, history

    def _kfac_compile_signature(
        self,
        batch_signature,
        config,
        stage,
        *,
        use_gpinn,
        active_regions,
        eqn_weight_regions,
        idxgall,
        n_regions,
    ):
        net = self.model_config.network
        regions = sorted(self.model_config.regions, key=lambda region: region.index)
        damping_mode = "adaptive" if bool(jnp.isnan(jnp.asarray(config["damping"]))) else "fixed"
        return {
            "model": {
                "workflow": self.model_config.workflow,
                "network": {
                    "depth": int(net.depth),
                    "width": int(net.width),
                    "activation": int(net.activation),
                    "first_layer_scale": float(net.first_layer_scale),
                    "embedding": bool(net.embedding),
                    "embed_n": int(net.embed_n),
                    "embed_std": float(net.embed_std),
                },
                "per_region_networks": _json_ready(self.model_config.per_region_networks),
                "per_region_embeddings": _json_ready(self.model_config.per_region_embeddings),
                "anisotropic": bool(self.model_config.anisotropic),
                "regions": [
                    {"index": int(region.index), "kind": region.region_kind}
                    for region in regions
                ],
                "n_regions": int(n_regions),
            },
            "loss": {
                "name": self.loss_config.name,
                "matching": bool(self.loss_config.matching),
                "calving_front": bool(self.loss_config.calving_front),
                "use_gpinn": bool(use_gpinn),
                "active_regions": _json_ready(active_regions),
                "eqn_weight_regions": _json_ready(eqn_weight_regions),
                "idxgall": _json_ready(idxgall),
                "global_weights": _json_ready(self.loss_config.global_weights),
            },
            "training": {
                "adaptive_sampling": bool(stage.adaptive_sampling),
                "adaptive_sampling_burn_in": int(stage.adaptive_sampling_burn_in),
                "adaptive_sampling_period": int(stage.adaptive_sampling_period),
                "damping_mode": damping_mode,
            },
            "kfac_config": _json_ready(config),
            "batch_shape_signature": batch_signature,
            "jax_cache_env": _jax_cache_env_signature(),
        }

    def _effective_gpinn_enabled(self):
        if not self.loss_config.use_gpinn:
            return False
        weights = self.loss_config.global_weights
        weight = self.loss_config.gpinn_weight
        if weights is not None and "gpinn" in weights:
            weight = weights["gpinn"]
        return float(jnp.asarray(weight)) != 0.0

    def _xpinn_regions(self, raw_data):
        """Infer floating sub-regions when the user omits explicit regions."""

        if self.model_config.regions:
            return sorted(self.model_config.regions, key=lambda region: region.index)
        n_sub = len(raw_data["xd"][0])
        return [RegionConfig("floating", idx) for idx in range(n_sub)]

    def _validate_schemas(self, basal_mask):
        """Validate equation field contracts before expensive initialization."""

        schemas = []
        for basal in basal_mask:
            if basal:
                schemas.append(FieldSchema("grounded", ("u", "v", "h", "s", "mu", "C")))
            else:
                schemas.append(FieldSchema("floating", ("u", "v", "h", "mu"), ("s",)))
        validate_contracts(self.equation_config.name, schemas)

    def _adaptive_eval(self, params):
        """Build the residual evaluator used by RAD sampling."""

        state = self.state
        if state is None or self.model_config.workflow != "xpinn":
            return None

        scales_by_region = {
            idx: scale
            for idx, scale in zip(state.sub_region_indices, state.scales)
        }

        def eval_f(current_params, x, idx, basal):
            idx = int(idx)
            basal = bool(basal)
            scale = scales_by_region[idx]
            net = lambda z: state.solution[0](current_params, z, idx)
            point_eval = lambda z: gov_eqn_iso(net, z, scale, basal=basal)
            if getattr(x, "ndim", 0) == 1:
                return point_eval(x)
            residuals = []
            terms = []
            for start in range(0, x.shape[0], 2048):
                residual, term = jax.vmap(point_eval, in_axes=(0,))(x[start:start + 2048])
                residuals.append(residual)
                terms.append(term)
            return jnp.vstack(residuals), jnp.vstack(terms)

        return eval_f

    def _require_prepared(self):
        """Return prepared state or fail with the expected call order."""

        if self.state is None:
            raise RuntimeError("Call prepare() before fit(), predict(), or save().")
        return self.state
