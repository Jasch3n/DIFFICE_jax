from pathlib import Path
import argparse
import json
import os
import pickle
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffice-jax")

import jax
import numpy as np

import diffice_jax as djax
from tests.test_xpinn_joint_inversion_kfac import (
    C_REL_MAE_MIN_TRUTH,
    render_plot_cache,
    _plot_paths,
    _save_plot_cache,
)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_solver_dir(config, solver_dir):
    if solver_dir is not None:
        return solver_dir
    output_dir = config.artifacts.get("output_dir")
    if output_dir is None:
        raise ValueError("Pass --solver-dir or set artifacts.output_dir in the workflow config.")
    tag = config.artifacts.get("tag", config.name)
    output_dir = str(output_dir).format(tag=tag, name=config.name)
    path = Path(output_dir)
    return path if path.is_absolute() else config.base_dir / path


def _warn_if_saved_config_differs(config, solver_dir):
    saved_config_path = solver_dir / "config.json"
    if not saved_config_path.exists():
        return
    with open(saved_config_path, encoding="utf-8") as f:
        saved = json.load(f)
    requested_use_gpinn = bool(config.loss.get("use_gpinn", False))
    saved_use_gpinn = saved.get("loss", {}).get("use_gpinn")
    saved_optimizer_use_gpinn = (
        saved.get("training", {})
        .get("stages", [{}])[0]
        .get("optimizer", {})
        .get("parameters", {})
        .get("use_gpinn")
    )
    if saved_use_gpinn is None and saved_optimizer_use_gpinn is not None:
        saved_use_gpinn = bool(saved_optimizer_use_gpinn)
    if saved_use_gpinn is not None and bool(saved_use_gpinn) != requested_use_gpinn:
        warnings.warn(
            f"Saved solver config use_gpinn={saved_use_gpinn} differs from requested config "
            f"use_gpinn={requested_use_gpinn}. The rendered fields come from {solver_dir}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _region_velocity_truth(solver, idx, key):
    raw_data = solver.state.raw_data
    if key not in raw_data:
        return None
    data = solver.state.normalized_data[idx]
    idxval = np.asarray(jax.device_get(data[4][4][0]), dtype=int).reshape(-1)
    return np.asarray(raw_data[key][0, idx]).reshape(-1)[idxval]


def _field_region(x, y, truth, pred, min_truth=None):
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    mask = np.isfinite(truth)
    if min_truth is not None:
        mask &= np.abs(truth) > min_truth
    mismatch = np.full_like(pred, np.nan, dtype=float)
    mismatch[mask] = np.abs(pred[mask] - truth[mask]) / np.maximum(np.abs(truth[mask]), 1e-12)
    return dict(
        x=np.asarray(x).reshape(-1),
        y=np.asarray(y).reshape(-1),
        truth=truth.reshape(-1),
        pred=pred.reshape(-1),
        mismatch=mismatch.reshape(-1),
    )


def _value_region(x, y, value):
    return dict(x=np.asarray(x).reshape(-1), y=np.asarray(y).reshape(-1), value=np.asarray(value).reshape(-1))


def _prediction_plot_cache(solver, predictions, diagnostics, loss_history, tag, metadata):
    mu_regions = []
    c_regions = []
    residual_regions = {"x": [], "y": []}
    term_regions = {"term1_1": [], "term12_2": [], "term1_3": [], "term1_4": []}
    scale_by_idx = dict(zip(solver.state.sub_region_indices, solver.state.scales))

    for region in predictions["regions"]:
        idx = region["index"]
        x = np.asarray(jax.device_get(region["x"])).reshape(-1)
        y = np.asarray(jax.device_get(region["y"])).reshape(-1)
        mu_pred = np.asarray(jax.device_get(region["mu"])).reshape(-1)
        c_pred = np.asarray(jax.device_get(region["C"])).reshape(-1)
        mu_true = _region_velocity_truth(solver, idx, "mud")
        c_true = _region_velocity_truth(solver, idx, "alpha2d")
        if mu_true is not None:
            mu_regions.append(_field_region(x, y, mu_true, mu_pred))
        if c_true is not None and np.isfinite(c_pred).any():
            c_regions.append(_field_region(x, y, c_true, c_pred, min_truth=C_REL_MAE_MIN_TRUTH))

    for region in diagnostics["regions"]:
        idx = region["index"]
        x = np.asarray(jax.device_get(region["x"])).reshape(-1)
        y = np.asarray(jax.device_get(region["y"])).reshape(-1)
        residual_x = np.asarray(jax.device_get(region["residual_x"])).reshape(-1)
        residual_y = np.asarray(jax.device_get(region["residual_y"])).reshape(-1)
        terms = np.asarray(jax.device_get(region["raw_terms"]))
        if terms.shape[1] >= 8:
            gamma_mu = float(scale_by_idx[idx].dynamic_scale.gamma_mu)
            x_norm = terms[:, 2] / gamma_mu
            y_ref = np.column_stack((terms[:, 3:6], terms[:, 8:9]))
        else:
            x_norm = np.max(np.abs(terms[:, 0:3]), axis=1)
            y_ref = terms[:, 3:6]
        y_norm = np.max(np.abs(y_ref), axis=1)
        residual_regions["x"].append(_value_region(x, y, residual_x / (x_norm + 1e-10)))
        residual_regions["y"].append(_value_region(x, y, residual_y / (10.0 * (y_norm + 1e-10))))

        term1_3 = terms[:, 2]
        denom = np.where(np.abs(term1_3) < 1e-10, np.where(term1_3 < 0.0, -1e-10, 1e-10), term1_3)
        term_regions["term1_1"].append(_value_region(x, y, terms[:, 0] / denom))
        term_regions["term12_2"].append(_value_region(x, y, terms[:, 1] / denom))
        term_regions["term1_3"].append(_value_region(x, y, term1_3 / denom))
        if terms.shape[1] >= 8:
            term_regions["term1_4"].append(_value_region(x, y, terms[:, 7] / denom))

    return dict(
        tag=tag,
        mu_regions=mu_regions,
        c_regions=c_regions,
        residual_regions=residual_regions,
        term_regions=term_regions,
        loss_history=np.asarray(loss_history, dtype=float),
        metadata=metadata,
    )


def render_from_saved_solver(config_path, solver_dir, tag):
    config = djax.load_workflow_config(config_path)
    solver_dir = _resolve_solver_dir(config, solver_dir)
    _warn_if_saved_config_differs(config, solver_dir)
    solver = djax.build_solver_from_config(config).prepare().load_params(solver_dir / "params.pkl")
    loss_history = _load_pickle(solver_dir / "loss_history.pkl")

    start = time.perf_counter()
    predictions = solver.predict()
    diagnostics = solver.predict_equation_diagnostics(points="velocity")
    predict_seconds = time.perf_counter() - start

    paths = _plot_paths(tag)
    metadata = dict(
        optimizer="KFAC",
        requested_iterations=int(config.training["stages"][0]["iterations"]),
        start_iteration=0,
        final_iteration=int(np.asarray(loss_history, dtype=float)[-1, 0]),
        trained_iterations=int(np.asarray(loss_history, dtype=float)[-1, 0]),
        elapsed_seconds=np.nan,
        prediction_seconds=predict_seconds,
        initial_objective=float(np.asarray(loss_history, dtype=float)[0, 1]),
        final_objective=float(np.asarray(loss_history, dtype=float)[-1, 1]),
        loss_history=np.asarray(loss_history, dtype=float),
        data_path=str(solver.data_config.source),
        sample_count=np.nan,
        requested_interface_points=np.nan,
        interface_point_counts=np.asarray([]),
        interface_collocation=config.data["interface_collocation"]["sample_count"],
        use_gpinn=bool(config.loss.get("use_gpinn", False)),
        legacy_kfac_eval=False,
        adaptive_sampling=bool(config.training["stages"][0].get("adaptive_sampling", False)),
        depth=int(config.model["network"]["depth"]),
        width=int(config.model["network"]["width"]),
        solver_dir=str(solver_dir),
    )
    _save_plot_cache(paths["cache"], _prediction_plot_cache(solver, predictions, diagnostics, loss_history, tag, metadata))
    return render_plot_cache(paths["cache"], tag=tag)


def main():
    parser = argparse.ArgumentParser(description="Render XPINN KFAC plots from saved DIFFICESolver params.")
    parser.add_argument("--config", type=Path, default=Path("examples/configs/xpinn_joint_flatbed_kfac.yaml"))
    parser.add_argument("--solver-dir", type=Path, default=None)
    parser.add_argument("--tag", default="solver_predict")
    args = parser.parse_args()

    output, fields, loss, residuals, ratios, cache, mu_rel_mae, c_rel_mae = render_from_saved_solver(
        args.config,
        args.solver_dir,
        args.tag,
    )
    print(f"PLOT_OUTPUT={output}")
    print(f"FIELD_COMPARISON={fields}")
    print(f"LOSS_CURVE={loss}")
    print(f"EQUATION_RESIDUALS={residuals}")
    print(f"X_TERM_RATIOS={ratios}")
    print(f"PLOT_CACHE={cache}")
    print(f"MU_REL_MAE={mu_rel_mae:.6f}")
    print(f"C_REL_MAE={c_rel_mae:.6f}")


if __name__ == "__main__":
    main()
