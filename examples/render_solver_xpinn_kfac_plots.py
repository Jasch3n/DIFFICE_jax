from pathlib import Path
import argparse
import json
import os
import pickle
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffice-jax")

import jax
import matplotlib
import numpy as np

import diffice_jax as djax
from diffice_jax.plotting import tripcolor_scattered

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ARTIFACT_PREFIX = "test_xpinn_joint_inversion_flatbed"
C_REL_MAE_MIN_TRUTH = 1e-3
DEFAULT_FONT_FAMILY = "Noto Sans"
DEFAULT_FONT_PATH = Path.home() / "Library" / "Fonts" / "NotoSans-Regular.ttf"
DEFAULT_FONT_DIR = Path.home() / "Library" / "Fonts"
DEFAULT_FONT_FILES = (
    "NotoSans-Light.ttf",
    "NotoSans-Thin.ttf",
    "NotoSans-ExtraThin.ttf",
    "NotoSans-Regular.ttf",
    "NotoSans-Italic.ttf",
    "NotoSans-Bold.ttf",
    "NotoSans-BoldItalic.ttf",
    "NotoSans-ExtraBold.ttf",
    "NotoSans-ExtraBoldItalic.ttf",
)


def configure_plot_font(font_family=DEFAULT_FONT_FAMILY, font_path=DEFAULT_FONT_PATH):
    font_path = Path(font_path).expanduser() if font_path else None
    for font_name in DEFAULT_FONT_FILES:
        candidate = DEFAULT_FONT_DIR / font_name
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
    if font_path is not None and font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans", "Ubuntu"]
    plt.rcParams["font.weight"] = 300
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelweight"] = 300
    plt.rcParams["axes.titleweight"] = 700
    plt.rcParams["figure.titleweight"] = 700
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font_family
    plt.rcParams["mathtext.it"] = f"{font_family}:italic"
    plt.rcParams["mathtext.bf"] = f"{font_family}:bold"
    plt.rcParams["mathtext.default"] = "regular"


configure_plot_font()


def _set_axis_font_weight(
    ax,
    label_weight=plt.rcParams["axes.labelweight"],
    title_weight=plt.rcParams["axes.titleweight"],
):
    ax.xaxis.label.set_fontweight(label_weight)
    ax.yaxis.label.set_fontweight(label_weight)
    ax.title.set_fontweight(title_weight)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(label_weight)


def _field_stats(pred, truth, min_truth=None):
    pred = np.asarray(pred).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(truth)
    if min_truth is not None:
        mask &= np.abs(truth) > min_truth
    pred = pred[mask]
    truth = truth[mask]
    if truth.size == 0:
        return np.nan
    rel_mae = np.mean(np.abs(pred - truth)) / np.maximum(np.mean(np.abs(truth)), 1e-12)
    return float(rel_mae)


def _concat_region_values(regions, key):
    if not regions:
        return np.asarray([])
    return np.concatenate([np.asarray(region[key]).reshape(-1) for region in regions])


def _domain_bounds(regions):
    x = _concat_region_values(regions, "x") / 1e3
    y = _concat_region_values(regions, "y") / 1e3
    return (
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )


def _add_colorbar(fig, ax, image, label):
    cbar = fig.colorbar(image, ax=ax, fraction=0.015, pad=0.05)
    cbar.set_label(label)
    _set_axis_font_weight(cbar.ax)


def _tripcolor_regions(ax, regions, key, title, cmap, vmin, vmax):
    image = None
    for region in regions:
        values = np.asarray(region[key]).reshape(-1)
        mask = np.isfinite(region["x"]) & np.isfinite(region["y"]) & np.isfinite(values)
        if np.count_nonzero(mask) < 3:
            continue
        xy = np.column_stack([np.asarray(region["x"])[mask], np.asarray(region["y"])[mask]])
        _, keep = np.unique(xy, axis=0, return_index=True)
        image = tripcolor_scattered(
            xy[keep, 0],
            xy[keep, 1],
            values[mask][keep],
            ax=ax,
            coordinate_scale=1e-3,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    ax.set_title(title)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    _set_axis_font_weight(ax)
    return image


def save_inversion_comparison(path, mu_regions, c_regions):
    path.parent.mkdir(parents=True, exist_ok=True)
    mu_true = _concat_region_values(mu_regions, "truth")
    mu_pred = _concat_region_values(mu_regions, "pred")
    c_true = _concat_region_values(c_regions, "truth")
    c_pred = _concat_region_values(c_regions, "pred")
    c_mismatch = _concat_region_values(c_regions, "mismatch")
    mu_rel_mae = _field_stats(mu_pred, mu_true)
    c_rel_mae = _field_stats(c_pred, c_true, min_truth=C_REL_MAE_MIN_TRUTH)

    mu_vmin = np.nanpercentile(mu_true, 5)
    mu_vmax = np.nanpercentile(mu_true, 95)
    c_vmin = np.nanpercentile(c_true, 5) if c_true.size else np.nan
    c_vmax = np.nanpercentile(c_true, 80) if c_true.size else np.nan
    c_mismatch_vmax = 0.15 if c_mismatch.size else np.nan
    x_min, x_max, y_min, y_max = _domain_bounds(mu_regions)

    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=True)
    panels = [
        (axs[0, 0], mu_regions, "truth", "ground truth viscosity $\\mu$", "viridis", mu_vmin, mu_vmax, "Pa s"),
        (axs[0, 1], c_regions, "truth", "ground truth basal friction $C$", "cool", c_vmin, c_vmax, "Pa s/m"),
        (axs[1, 0], mu_regions, "pred", "inferred viscosity $\\mu$", "viridis", mu_vmin, mu_vmax, "Pa s"),
        (axs[1, 1], c_regions, "pred", "inferred basal friction $C$", "cool", c_vmin, c_vmax, "Pa s/m"),
        (axs[2, 0], mu_regions, "mismatch", f"relative absolute error (mean = {100.0 * mu_rel_mae:.2f}%)", "magma", 0.0, 0.15, "|error| / |truth|"),
        (axs[2, 1], c_regions, "mismatch", f"relative absolute error (mean = {100.0 * c_rel_mae:.2f}%)", "magma", 0.0, c_mismatch_vmax, "|error| / |truth|"),
    ]
    for ax, regions, key, title, cmap, lo, hi, label in panels:
        image = _tripcolor_regions(ax, regions, key, title, cmap, lo, hi)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_box_aspect(1 / 3)
        if image is not None:
            _add_colorbar(fig, ax, image, label)
    fig.suptitle("Synthetic XPINN Joint Inversion (K-FAC Optimizer)", fontsize=15, fontweight=800)
    plt.tight_layout()
    fig.canvas.draw()
    for ax in fig.axes:
        _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return mu_rel_mae, c_rel_mae


def _signed_percentile_limit(regions, key="value", percentile=98, floor=1e-12):
    values = _concat_region_values(regions, key)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return float(max(np.nanpercentile(np.abs(values), percentile), floor))


def save_equation_residuals(path, residual_regions):
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = _domain_bounds(residual_regions["x"])
    x_vmax = 0.1
    y_vmax = 0.1
    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, sharey=True)
    panels = [
        (axs[0], residual_regions["x"], "relative x-equation residual", "coolwarm", -x_vmax, x_vmax, "normalized residual"),
        (axs[1], residual_regions["y"], "relative y-equation residual", "coolwarm", -y_vmax, y_vmax, "normalized residual"),
    ]
    for ax, regions, title, cmap, lo, hi, label in panels:
        image = _tripcolor_regions(ax, regions, "value", title, cmap, lo, hi)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_box_aspect(1 / 3)
        if image is not None:
            _add_colorbar(fig, ax, image, label)
    fig.suptitle("Synthetic XPINN Equations Residuals", fontsize=15, fontweight=800)
    plt.tight_layout()
    for ax in fig.axes:
        _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_x_equation_term_ratios(path, term_regions, bounds_regions):
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = _domain_bounds(bounds_regions)
    fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True, sharey=True)
    panels = [
        (axs[0], term_regions["term1_1"], r"viscous gradient x", 1.0),
        (axs[1], term_regions["term12_2"], r"viscous gradient y", 1.0),
        (axs[2], term_regions["term1_4"], r"basal friction $\tau_{bx}$", 1.0),
    ]
    for ax, regions, title, bound in panels:
        _signed_percentile_limit(regions)
        image = _tripcolor_regions(ax, regions, "value", title, "coolwarm", -bound, bound)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_box_aspect(1 / 3)
        if image is not None:
            _add_colorbar(fig, ax, image, "signed ratio")
    fig.suptitle(r"x-equation terms relative to driving stress", fontsize=15, fontweight=800)
    plt.tight_layout()
    for ax in fig.axes:
        _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_loss_curve(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(history, dtype=float)
    if history.size == 0:
        return
    order = np.argsort(history[:, 0])
    steps = history[order, 0]
    losses = history[order, 1]
    mask = np.isfinite(steps) & np.isfinite(losses) & (losses > 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.semilogy(steps[mask], losses[mask], color="black", linewidth=1.8)
    ax.set_xlabel("KFAC iteration")
    ax.set_ylabel("Residual objective")
    ax.set_title("Synthetic XPINN joint-inversion KFAC loss")
    _set_axis_font_weight(ax)
    ax.grid(True, which="both", alpha=0.25)
    fig.canvas.draw()
    _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_plot_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _plot_paths(plot_dir, tag):
    plot_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        output=plot_dir / "data.npz",
        fields=plot_dir / "fields.png",
        loss=plot_dir / "loss.png",
        equation_residuals=plot_dir / "equation_residuals.png",
        x_term_ratios=plot_dir / "x_term_ratio.png",
        cache=plot_dir / "plot_cache.pkl",
    )


def _render_plot_cache(cache_path, plot_dir, tag=None):
    cache_path = Path(cache_path)
    cache = _load_pickle(cache_path)
    tag = tag or cache["tag"]
    paths = _plot_paths(plot_dir, tag)
    mu_rel_mae, c_rel_mae = save_inversion_comparison(
        paths["fields"],
        cache["mu_regions"],
        cache["c_regions"],
    )
    save_equation_residuals(paths["equation_residuals"], cache["residual_regions"])
    save_x_equation_term_ratios(paths["x_term_ratios"], cache["term_regions"], cache["residual_regions"]["x"])
    save_loss_curve(paths["loss"], cache["loss_history"])
    np.savez(
        paths["output"],
        **cache["metadata"],
        mu_rel_mae=mu_rel_mae,
        c_rel_mae=c_rel_mae,
        target_c_rel_mae=np.nan,
        field_comparison_path=str(paths["fields"]),
        equation_residual_path=str(paths["equation_residuals"]),
        x_term_ratio_path=str(paths["x_term_ratios"]),
        loss_curve_path=str(paths["loss"]),
        plot_cache_path=str(cache_path),
    )
    return paths["output"], paths["fields"], paths["loss"], paths["equation_residuals"], paths["x_term_ratios"], cache_path, mu_rel_mae, c_rel_mae


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

    plot_dir = solver_dir / "plots"
    paths = _plot_paths(plot_dir, tag)
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
    return _render_plot_cache(paths["cache"], plot_dir, tag=tag)


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
