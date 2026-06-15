from pathlib import Path
import argparse
import os
import pickle
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffice-jax")

import jax
from jax import random
import jax.numpy as jnp
import matplotlib
import numpy as np

import diffice_jax as djax
from diffice_jax import KfacOptimizer
from diffice_jax.core.solver import (
    attach_gpinn_interface_collocation,
    limit_xpinn_batch,
    replace_xpinn_collocation,
)
from diffice_jax.data.xpinns import sampling as xpinn_sampling
from tests.test_xpinn_regression import xpinn_regression as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FIGURE_DIR = Path(__file__).parent / "figures"
DATA_PATH = Path(__file__).parent / "test_xpinn_regression" / "flatbed_data_xpinns_regression_test.mat"

DEFAULT_DEPTH = 6
DEFAULT_WIDTH = 30
DEFAULT_SAMPLE_COUNT = 1028
DEFAULT_INTERFACE_POINTS = None
DEFAULT_CALVING_FRONT_POINTS = 500
DEFAULT_INTERFACE_COLLOCATION = 500
DEFAULT_ADAPTIVE_SAMPLING = True
C_REL_MAE_MIN_TRUTH = 1e-3
ARTIFACT_PREFIX = "test_xpinn_joint_inversion_flatbed"
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
    plt.rcParams["font.sans-serif"] = [font_family, 'DejaVu Sans', 'Ubuntu']
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


def _artifact_dir(tag):
    path = FIGURE_DIR / f"joint_inversion__{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_path(tag, suffix):
    return _artifact_dir(tag) / f"{ARTIFACT_PREFIX}_{tag}{suffix}"


def _set_axis_font_weight(ax, 
                          label_weight=plt.rcParams["axes.labelweight"], 
                          title_weight=plt.rcParams["axes.titleweight"]):
    ax.xaxis.label.set_fontweight(label_weight)
    ax.yaxis.label.set_fontweight(label_weight)
    ax.title.set_fontweight(title_weight)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(label_weight)


def kfac_config():
    return dict(
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


def _limit_rows(x, n):
    if n is None or x.shape[0] <= n:
        return x
    return x[:n]


def _limit_batch(batch, interface_points=None, calving_front_points=None):
    return limit_xpinn_batch(batch, interface_points)


def test_limit_batch_keeps_all_interface_points_by_default():
    x_md = jnp.arange(40.0).reshape(10, 4)
    batch = dict(
        md=[[x_md]],
        ct=[[jnp.zeros((4, 2))], [jnp.zeros((4, 2))]],
    )

    limited = _limit_batch(batch, calving_front_points=2)

    assert limited["md"][0][0].shape[0] == 10
    assert limited["ct"][0][0].shape[0] == 4
    assert limited["ct"][1][0].shape[0] == 4


def test_limit_batch_can_limit_interface_points_for_benchmarks():
    x_md = jnp.arange(40.0).reshape(10, 4)
    batch = dict(
        md=[[x_md]],
        ct=[[jnp.zeros((4, 2))], [jnp.zeros((4, 2))]],
    )

    limited = _limit_batch(batch, interface_points=3)

    assert limited["md"][0][0].shape[0] == 3


def test_gpinn_interface_collocation_uses_collocation_tail():
    old_n = xpinn_sampling.N_INTERFACE_COLLOCATION
    xpinn_sampling.N_INTERFACE_COLLOCATION = 2
    try:
        col0 = jnp.arange(12.0).reshape(6, 2)
        col1 = jnp.arange(100.0, 112.0).reshape(6, 2)
        md = jnp.arange(200.0, 216.0).reshape(4, 4)
        batch = dict(col=[[col0, col1]], md=[[md]])

        updated = _attach_gpinn_interface_collocation(batch, 2)

        assert jnp.allclose(updated["gpinn_col"][0][0], col0[-2:])
        assert jnp.allclose(updated["gpinn_col"][0][1], col1[-2:])
    finally:
        xpinn_sampling.N_INTERFACE_COLLOCATION = old_n


def _attach_gpinn_interface_collocation(batch, n_regions):
    return attach_gpinn_interface_collocation(batch, n_regions)


def _replace_collocation(batch, x_col, use_gpinn=False, n_regions=None):
    return replace_xpinn_collocation(batch, x_col, use_gpinn=use_gpinn, n_regions=n_regions)


def test_reused_adaptive_collocation_refreshes_gpinn_tail():
    old_n = xpinn_sampling.N_INTERFACE_COLLOCATION
    xpinn_sampling.N_INTERFACE_COLLOCATION = 2
    try:
        old_col = [
            jnp.arange(12.0).reshape(6, 2),
            jnp.arange(100.0, 112.0).reshape(6, 2),
        ]
        new_col = [
            jnp.arange(200.0, 212.0).reshape(6, 2),
            jnp.arange(300.0, 312.0).reshape(6, 2),
        ]
        batch = _attach_gpinn_interface_collocation(dict(col=[old_col]), 2)

        updated = _replace_collocation(batch, new_col, use_gpinn=True, n_regions=2)

        assert jnp.allclose(updated["gpinn_col"][0][0], new_col[0][-2:])
        assert jnp.allclose(updated["gpinn_col"][0][1], new_col[1][-2:])
    finally:
        xpinn_sampling.N_INTERFACE_COLLOCATION = old_n


def _prepare_flatbed_xpinn(
    seed,
    depth,
    width,
    sample_count,
    interface_collocation,
    use_gpinn=False,
    adaptive_sampling=DEFAULT_ADAPTIVE_SAMPLING,
):
    xr.set_train_mode("full")
    xr.USE_GPINN = bool(use_gpinn)
    xr.USE_GPINN_IN_KFAC = bool(use_gpinn)
    xr.USE_ADAPTIVE_SAMPLING = bool(adaptive_sampling)
    xr.USE_REGION_TERM_BALANCING = False
    xr.REGRESSION_N_PT_BY_NSUB[2] = [
        [sample_count, sample_count],
        [sample_count, sample_count],
        [sample_count, sample_count],
    ]
    xr.NETWORK_CONFIG = [
        xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
        xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
    ]
    xpinn_sampling.N_INTERFACE_LIBRARY = interface_collocation
    xpinn_sampling.N_INTERFACE_COLLOCATION = interface_collocation

    key = random.PRNGKey(seed)
    data_output = xr.load_data(str(DATA_PATH))
    key, xpinn_output = xr.initialize_xpinn(key, data_output)
    key, loss_output = xr.initialize_loss(key, data_output, xpinn_output)
    return key, data_output, xpinn_output, loss_output


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


def _collect_inversion_fields(params, data_output, xpinn_output):
    mu_regions = []
    c_regions = []
    for idx in data_output.idxgall:
        X = data_output.data_all[idx][0][0]
        truth = data_output.data_all[idx][1]
        raw = data_output.data_all[idx][4][3]
        idxval = np.asarray(data_output.data_all[idx][4][4][0], dtype=int).reshape(-1)
        out = xpinn_output.sol_NN[0](params, X, idx)
        scale = data_output.scale[idx].dynamic_scale
        x = np.asarray(raw[0]).reshape(-1)[idxval]
        y = np.asarray(raw[1]).reshape(-1)[idxval]
        mu_pred = np.asarray(out[:, 4]).reshape(-1) * float(scale.mu0)
        mu_true = np.asarray(truth[3]).reshape(-1) * float(scale.mu0)
        mu_regions.append(dict(x=x, y=y, truth=mu_true, pred=mu_pred,
                               mismatch=np.abs(mu_pred - mu_true) / np.maximum(np.abs(mu_true), 1e-12)))
        if data_output.basal_mask[idx]:
            c_pred = np.asarray(out[:, 5]).reshape(-1) * float(scale.c0)
            c_true = np.asarray(truth[4]).reshape(-1) * float(scale.c0)
            c_mask = np.abs(c_true) > C_REL_MAE_MIN_TRUTH
            c_mismatch = np.full_like(c_true, np.nan, dtype=float)
            c_mismatch[c_mask] = np.abs(c_pred[c_mask] - c_true[c_mask]) / np.maximum(np.abs(c_true[c_mask]), 1e-12)
            c_regions.append(dict(x=x, y=y, truth=c_true, pred=c_pred,
                                  mismatch=c_mismatch))
    return mu_regions, c_regions


def _collect_equation_diagnostics(params, data_output, xpinn_output, eps=1e-10):
    residual_regions = {"x": [], "y": []}
    term_regions = {"term1_1": [], "term12_2": [], "term1_3": [], "term1_4": []}
    for idx in data_output.idxgall:
        X = data_output.data_all[idx][0][0]
        raw = data_output.data_all[idx][4][3]
        idxval = np.asarray(data_output.data_all[idx][4][4][0], dtype=int).reshape(-1)
        f_eqn, terms = xpinn_output.eval_f(params, X, idx)
        f_eqn = np.asarray(f_eqn)
        terms = np.asarray(terms)
        x = np.asarray(raw[0]).reshape(-1)[idxval]
        y = np.asarray(raw[1]).reshape(-1)[idxval]

        if terms.shape[1] >= 8:
            gamma_mu = float(data_output.scale[idx].dynamic_scale.gamma_mu)
            x_norm = terms[:, 2] / gamma_mu
            y_ref = np.column_stack((terms[:, 3:6], terms[:, 8:9]))
        else:
            x_norm = np.max(np.abs(terms[:, 0:3]), axis=1)
            y_ref = terms[:, 3:6]
        y_norm = np.max(np.abs(y_ref), axis=1)
        residual_regions["x"].append(dict(x=x, y=y, value=f_eqn[:, 0] / (x_norm + eps)))
        residual_regions["y"].append(dict(x=x, y=y, value=f_eqn[:, 1] / (10.0 * (y_norm + eps))))

        term1_3 = terms[:, 2]
        denom = np.where(np.abs(term1_3) < eps, np.where(term1_3 < 0.0, -eps, eps), term1_3)
        term_regions["term1_1"].append(dict(x=x, y=y, value=terms[:, 0] / denom))
        term_regions["term12_2"].append(dict(x=x, y=y, value=terms[:, 1] / denom))
        term_regions["term1_3"].append(dict(x=x, y=y, value=term1_3 / denom))
        if terms.shape[1] >= 8:
            term_regions["term1_4"].append(dict(x=x, y=y, value=terms[:, 7] / denom))
    return residual_regions, term_regions


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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
            image = ax.tripcolor(
                xy[keep, 0] / 1e3,
                xy[keep, 1] / 1e3,
                values[mask][keep],
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
    mu_mismatch = _concat_region_values(mu_regions, "mismatch")
    c_true = _concat_region_values(c_regions, "truth")
    c_pred = _concat_region_values(c_regions, "pred")
    c_mismatch = _concat_region_values(c_regions, "mismatch")
    mu_rel_mae = _field_stats(mu_pred, mu_true)
    c_rel_mae = _field_stats(c_pred, c_true, min_truth=C_REL_MAE_MIN_TRUTH)

    mu_vmin = np.nanpercentile(mu_true, 5)
    mu_vmax = np.nanpercentile(mu_true, 95)
    c_vmin = np.nanpercentile(c_true, 5) if c_true.size else np.nan
    c_vmax = np.nanpercentile(c_true, 80) if c_true.size else np.nan
    mu_mismatch_vmax = 0.15
    c_mismatch_vmax = 0.15 if c_mismatch.size else np.nan
    x_min, x_max, y_min, y_max = _domain_bounds(mu_regions)

    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=True)
    panels = [
        (axs[0, 0], mu_regions, "truth", "ground truth viscosity $\\mu$", "viridis", mu_vmin, mu_vmax, "Pa s"),
        (axs[0, 1], c_regions, "truth", "ground truth basal friction $C$", "cool", c_vmin, c_vmax, "Pa s/m"),
        (axs[1, 0], mu_regions, "pred", "inferred viscosity $\\mu$", "viridis", mu_vmin, mu_vmax, "Pa s"),
        (axs[1, 1], c_regions, "pred", "inferred basal friction $C$", "cool", c_vmin, c_vmax, "Pa s/m"),
        (axs[2, 0], mu_regions, "mismatch", f"relative absolute error (mean = {100.0 * mu_rel_mae:.2f}%)", "magma", 0.0, mu_mismatch_vmax, "|error| / |truth|"),
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
    # x_vmax = _signed_percentile_limit(residual_regions["x"])
    x_vmax = 0.1
    # y_vmax = _signed_percentile_limit(residual_regions["y"])
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
        vmax = _signed_percentile_limit(regions)
        image = _tripcolor_regions(ax, regions, "value", title, "coolwarm", -bound, bound)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_box_aspect(1 / 3)
        if image is not None:
            _add_colorbar(fig, ax, image, "signed ratio")
    fig.suptitle(r"x-equation terms relative to driving stress", fontsize=15, fontweight=800)
    plt.tight_layout()
    # fig.canvas.draw()
    for ax in fig.axes:
        _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_loss_curve(path, history, target_c_rel_mae=None):
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
    if target_c_rel_mae is not None:
        ax.text(
            0.98,
            0.95,
            f"C MAE target: {100.0 * target_c_rel_mae:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontweight=300,
        )
    fig.canvas.draw()
    _set_axis_font_weight(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_paths(tag):
    return dict(
        output=_artifact_path(tag, ".npz"),
        fields=_artifact_path(tag, "_fields.png"),
        loss=_artifact_path(tag, "_loss.png"),
        equation_residuals=_artifact_path(tag, "_equation_residuals.png"),
        x_term_ratios=_artifact_path(tag, "_x_term_ratios.png"),
        cache=_artifact_path(tag, "_plot_cache.pkl"),
    )


def _save_plot_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def _load_plot_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def render_plot_cache(cache_path, tag=None, target_c_rel_mae=None):
    cache_path = Path(cache_path)
    cache = _load_plot_cache(cache_path)
    tag = tag or cache["tag"]
    paths = _plot_paths(tag)
    mu_rel_mae, c_rel_mae = save_inversion_comparison(
        paths["fields"],
        cache["mu_regions"],
        cache["c_regions"],
    )
    save_equation_residuals(paths["equation_residuals"], cache["residual_regions"])
    save_x_equation_term_ratios(paths["x_term_ratios"], cache["term_regions"], cache["residual_regions"]["x"])
    save_loss_curve(paths["loss"], cache["loss_history"], target_c_rel_mae)
    np.savez(
        paths["output"],
        **cache["metadata"],
        mu_rel_mae=mu_rel_mae,
        c_rel_mae=c_rel_mae,
        target_c_rel_mae=np.nan if target_c_rel_mae is None else target_c_rel_mae,
        field_comparison_path=str(paths["fields"]),
        equation_residual_path=str(paths["equation_residuals"]),
        x_term_ratio_path=str(paths["x_term_ratios"]),
        loss_curve_path=str(paths["loss"]),
        plot_cache_path=str(cache_path),
    )
    return paths["output"], paths["fields"], paths["loss"], paths["equation_residuals"], paths["x_term_ratios"], cache_path, mu_rel_mae, c_rel_mae


def plot_checkpoint(checkpoint_path, tag=None, target_c_rel_mae=None):
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    config = checkpoint["config"]
    _, data_output, xpinn_output, _ = _prepare_flatbed_xpinn(
        config["seed"],
        config["depth"],
        config["width"],
        config["sample_count"],
        config["interface_collocation"],
        config["use_gpinn"],
        config["adaptive_sampling"],
    )
    tag = tag or f"step_{int(checkpoint['step'])}"
    paths = _plot_paths(tag)
    mu_regions, c_regions = _collect_inversion_fields(checkpoint["params"], data_output, xpinn_output)
    residual_regions, term_regions = _collect_equation_diagnostics(checkpoint["params"], data_output, xpinn_output)
    metadata = dict(
        optimizer="KFAC",
        requested_iterations=int(checkpoint["step"]),
        start_iteration=np.nan,
        final_iteration=int(checkpoint["step"]),
        trained_iterations=np.nan,
        elapsed_seconds=np.nan,
        initial_objective=float(np.asarray(checkpoint["loss_history"])[0, 1]),
        final_objective=float(np.asarray(checkpoint["loss_history"])[-1, 1]),
        loss_history=np.asarray(checkpoint["loss_history"], dtype=float),
        data_path=str(DATA_PATH),
        sample_count=config["sample_count"],
        requested_interface_points=np.nan if config["requested_interface_points"] is None else config["requested_interface_points"],
        interface_point_counts=config["interface_point_counts"],
        calving_front_points=config["calving_front_points"],
        interface_collocation=config["interface_collocation"],
        use_gpinn=config["use_gpinn"],
        legacy_kfac_eval=config["legacy_kfac_eval"],
        adaptive_sampling=config["adaptive_sampling"],
        depth=config["depth"],
        width=config["width"],
        checkpoint_path=str(checkpoint_path),
    )
    _save_plot_cache(
        paths["cache"],
        dict(
            tag=tag,
            mu_regions=mu_regions,
            c_regions=c_regions,
            residual_regions=residual_regions,
            term_regions=term_regions,
            loss_history=np.asarray(checkpoint["loss_history"], dtype=float),
            metadata=metadata,
        ),
    )
    return render_plot_cache(paths["cache"], tag=tag, target_c_rel_mae=target_c_rel_mae)


def _inversion_mae(params, data_output, xpinn_output):
    mu_regions, c_regions = _collect_inversion_fields(params, data_output, xpinn_output)
    mu_true = _concat_region_values(mu_regions, "truth")
    mu_pred = _concat_region_values(mu_regions, "pred")
    c_true = _concat_region_values(c_regions, "truth")
    c_pred = _concat_region_values(c_regions, "pred")
    return _field_stats(mu_pred, mu_true), _field_stats(c_pred, c_true, min_truth=C_REL_MAE_MIN_TRUTH)


def _save_checkpoint(path, step, params, opt_state, damping, key, history, mu_rel_mae, c_rel_mae, config, x_col_mem=None, adapted=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(
        step=step,
        params=jax.device_get(params),
        opt_state=jax.device_get(opt_state),
        damping=jax.device_get(damping),
        key=jax.device_get(key),
        loss_history=np.asarray(history, dtype=float),
        mu_rel_mae=mu_rel_mae,
        c_rel_mae=c_rel_mae,
        config=config,
        x_col_mem=None if x_col_mem is None else jax.device_get(x_col_mem),
        adapted=adapted,
    )
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def run_kfac_experiment(
    iterations=1500,
    seed=8132002,
    tag=None,
    log_rate=100,
    depth=DEFAULT_DEPTH,
    width=DEFAULT_WIDTH,
    sample_count=DEFAULT_SAMPLE_COUNT,
    interface_points=DEFAULT_INTERFACE_POINTS,
    calving_front_points=DEFAULT_CALVING_FRONT_POINTS,
    interface_collocation=DEFAULT_INTERFACE_COLLOCATION,
    use_gpinn=False,
    target_c_rel_mae=None,
    continue_chunk=500,
    max_iterations=None,
    checkpoint_dir=None,
    resume_checkpoint=None,
    legacy_kfac_eval=False,
    adaptive_sampling=DEFAULT_ADAPTIVE_SAMPLING,
):
    if jax.default_backend().lower() == "metal":
        raise RuntimeError("KFAC is unavailable on the JAX Metal backend. Run with JAX_PLATFORM_NAME=cpu.")

    from kfac_jax import loss_functions as kfac_loss_functions

    tag = tag or f"{iterations}_kfac"
    key, data_output, xpinn_output, loss_output = _prepare_flatbed_xpinn(
        seed,
        depth,
        width,
        sample_count,
        interface_collocation,
        use_gpinn,
        adaptive_sampling,
    )
    lossf = loss_output.loss_f
    dataf = loss_output.data_f
    params = dict(xpinn_output.params)

    def use_kfac_eval():
        return hasattr(lossf, "kfac_eval") and not legacy_kfac_eval

    def residual_vector(current_params, batch):
        return lossf.kfac_residuals(current_params, batch)

    def objective_value(current_params, batch):
        if use_kfac_eval():
            return lossf.kfac_eval(current_params, batch)[0]
        residuals = residual_vector(current_params, batch)
        return jnp.sum(jnp.square(residuals)) / lossf.lref

    def kfac_lossf(current_params, batch):
        if use_kfac_eval():
            loss_n, loss_info, _, raw_residuals = lossf.kfac_eval(current_params, batch)
            residuals = raw_residuals / jnp.sqrt(lossf.lref)
        else:
            _, loss_info, _ = lossf(current_params, batch)
            residuals = residual_vector(current_params, batch) / jnp.sqrt(lossf.lref)
            loss_n = jnp.sum(jnp.square(residuals))
        kfac_loss_functions.register_squared_error_loss(
            residuals,
            targets=jnp.zeros_like(residuals),
        )
        return loss_n, loss_info

    config = kfac_config()
    optim = KfacOptimizer(loss_fn=kfac_lossf, **config).get_optimizer()
    key, init_key = random.split(key)
    def sampled_batch(batch_key, step, **sample_kwargs):
        batch = dataf(batch_key, **sample_kwargs)
        batch = _limit_batch(batch, interface_points, calving_front_points)
        if use_gpinn:
            batch = _attach_gpinn_interface_collocation(batch, len(data_output.idxgall))
        return xr.attach_loss_weights(batch, step, data_output.idxgall)

    init_batch = sampled_batch(init_key, 0)
    interface_point_counts = np.asarray([x.shape[0] for x in init_batch["md"][0]], dtype=int)
    opt_state = optim.init(params, init_key, init_batch)

    initial_objective = float(objective_value(params, init_batch))
    history = [(0, initial_objective)]
    damping = config["initial_damping"]
    x_col_mem = None
    adapted = False
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _artifact_dir(tag) / "checkpoints"
    checkpoint_path = None
    final_step = 0

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
    while True:
        segment_end = iterations if final_step < iterations else final_step + continue_chunk
        if max_iterations is not None:
            segment_end = min(segment_end, max_iterations)
        for step in range(final_step, segment_end):
            key, step_key, data_key = random.split(key, 3)
            run_rad = (
                adaptive_sampling
                and (step + 1) % xr.ADAPT_PERIOD == 0
                and (step + 1) > xr.ADAPT_BURNIN
            )
            if run_rad:
                batch = sampled_batch(
                    data_key,
                    step,
                    eval_adaptive=True,
                    eval_f=lambda x, idx, basal: xpinn_output.eval_f(params, x, idx),
                )
                x_col_mem = batch["col"][0]
                adapted = True
            elif adaptive_sampling and adapted:
                batch = sampled_batch(data_key, step, eval_adaptive=False)
                batch = _replace_collocation(
                    batch,
                    x_col_mem,
                    use_gpinn=use_gpinn,
                    n_regions=len(data_output.idxgall),
                )
            else:
                batch = sampled_batch(data_key, step)
            params, opt_state, stats = optim.step(
                params,
                opt_state,
                step_key,
                batch=batch,
                damping=damping,
                global_step_int=step,
            )
            if damping > config["min_damping"]:
                damping *= config["damping_adaptation_decay"]
            if (step + 1) % log_rate == 0 or step == 0 or step + 1 == segment_end:
                current_objective = float(objective_value(params, batch))
                history.append((step + 1, current_objective))
                loss_info = stats["aux"]
                print(
                    f"KFAC step {step + 1}: objective={current_objective:.4e} "
                    f"scalar_loss={float(loss_info[xr.LOSS_INFO_TOTAL_IDX]):.4e} "
                    f"damping={float(stats['damping']):.2e}",
                    flush=True,
                )

        final_step = segment_end
        mu_rel_mae, c_rel_mae = _inversion_mae(params, data_output, xpinn_output)
        checkpoint_path = checkpoint_dir / f"KFAC_step_{final_step}.pkl"
        _save_checkpoint(
            checkpoint_path,
            final_step,
            params,
            opt_state,
            damping,
            key,
            history,
            mu_rel_mae,
            c_rel_mae,
            dict(
                seed=seed,
                depth=depth,
                width=width,
                sample_count=sample_count,
                requested_interface_points=interface_points,
                interface_point_counts=interface_point_counts,
                calving_front_points=calving_front_points,
                interface_collocation=interface_collocation,
                use_gpinn=use_gpinn,
                legacy_kfac_eval=legacy_kfac_eval,
                adaptive_sampling=adaptive_sampling,
                resume_checkpoint="" if resume_checkpoint is None else str(resume_checkpoint),
            ),
            x_col_mem=x_col_mem,
            adapted=adapted,
        )
        print(
            f"KFAC checkpoint step {final_step}: C_REL_MAE={c_rel_mae:.6f} "
            f"MU_REL_MAE={mu_rel_mae:.6f} path={checkpoint_path}",
            flush=True,
        )
        if target_c_rel_mae is None or c_rel_mae < target_c_rel_mae:
            break
        if max_iterations is not None and final_step >= max_iterations:
            break

    elapsed = time.perf_counter() - start
    final_batch = sampled_batch(key, final_step)
    final_objective = float(objective_value(params, final_batch))
    history.append((final_step, final_objective))

    output_path = _artifact_path(tag, ".npz")
    comparison_path = _artifact_path(tag, "_fields.png")
    loss_curve_path = _artifact_path(tag, "_loss.png")
    equation_residual_path = _artifact_path(tag, "_equation_residuals.png")
    x_term_ratio_path = _artifact_path(tag, "_x_term_ratios.png")
    mu_regions, c_regions = _collect_inversion_fields(params, data_output, xpinn_output)
    mu_rel_mae, c_rel_mae = save_inversion_comparison(
        comparison_path,
        mu_regions,
        c_regions,
    )
    residual_regions, term_regions = _collect_equation_diagnostics(params, data_output, xpinn_output)
    save_equation_residuals(equation_residual_path, residual_regions)
    save_x_equation_term_ratios(x_term_ratio_path, term_regions, residual_regions["x"])
    save_loss_curve(loss_curve_path, history, target_c_rel_mae)
    np.savez(
        output_path,
        optimizer="KFAC",
        requested_iterations=iterations,
        start_iteration=start_step,
        final_iteration=final_step,
        trained_iterations=final_step - start_step,
        elapsed_seconds=elapsed,
        initial_objective=initial_objective,
        final_objective=final_objective,
        loss_history=np.asarray(history, dtype=float),
        data_path=str(DATA_PATH),
        sample_count=sample_count,
        requested_interface_points=np.nan if interface_points is None else interface_points,
        interface_point_counts=interface_point_counts,
        calving_front_points=calving_front_points,
        interface_collocation=interface_collocation,
        use_gpinn=use_gpinn,
        legacy_kfac_eval=legacy_kfac_eval,
        adaptive_sampling=adaptive_sampling,
        depth=depth,
        width=width,
        mu_rel_mae=mu_rel_mae,
        c_rel_mae=c_rel_mae,
        target_c_rel_mae=np.nan if target_c_rel_mae is None else target_c_rel_mae,
        checkpoint_path="" if checkpoint_path is None else str(checkpoint_path),
        field_comparison_path=str(comparison_path),
        equation_residual_path=str(equation_residual_path),
        x_term_ratio_path=str(x_term_ratio_path),
        loss_curve_path=str(loss_curve_path),
    )
    return initial_objective, final_objective, elapsed, output_path, comparison_path, loss_curve_path, start_step, final_step, checkpoint_path, mu_rel_mae, c_rel_mae


def _configure_flatbed_globals(depth, width, sample_count, interface_collocation, use_gpinn, adaptive_sampling):
    xr.set_train_mode("full")
    xr.USE_GPINN = bool(use_gpinn)
    xr.USE_GPINN_IN_KFAC = bool(use_gpinn)
    xr.USE_ADAPTIVE_SAMPLING = bool(adaptive_sampling)
    xr.USE_REGION_TERM_BALANCING = False
    xr.REGRESSION_N_PT_BY_NSUB[2] = [
        [sample_count, sample_count],
        [sample_count, sample_count],
        [sample_count, sample_count],
    ]
    xr.NETWORK_CONFIG = [
        xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
        xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
    ]
    xpinn_sampling.N_INTERFACE_LIBRARY = interface_collocation
    xpinn_sampling.N_INTERFACE_COLLOCATION = interface_collocation


def _build_solver(seed, depth, width, sample_count, interface_collocation, use_gpinn, adaptive_sampling, optimizer_parameters):
    _configure_flatbed_globals(depth, width, sample_count, interface_collocation, use_gpinn, adaptive_sampling)
    sampling_counts = [
        [sample_count, sample_count],
        [sample_count, sample_count],
        [sample_count, sample_count],
    ]
    return djax.DIFFICESolver(
        data=djax.DataConfig(
            source=DATA_PATH,
            sampling_counts=sampling_counts,
            regression_workflow=True,
        ),
        model=djax.ModelConfig(
            workflow="xpinn",
            regions=[
                djax.RegionConfig("grounded", 0),
                djax.RegionConfig("floating", 1),
            ],
            network=djax.NetworkConfig(depth=depth, width=width),
            per_region_networks=[
                xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
                xr.net_arch(u=(depth, width), mu=(depth, width), c0=(depth, width)),
            ],
        ),
        loss=djax.LossConfig(
            name="regression",
            matching=True,
            calving_front=True,
            gpinn_weight=xr.GPINN_WEIGHT if use_gpinn else 0.0,
            active_regions=xr.ACTIVE_REGIONS,
        ),
        training=djax.TrainingConfig(
            stages=[
                djax.TrainingStage(
                    optimizer=djax.OptimizerConfig(
                        name="kfac",
                        learning_rate=None,
                        damping=jnp.nan,
                        parameters=optimizer_parameters,
                    ),
                    iterations=optimizer_parameters.pop("iterations"),
                    adaptive_sampling=adaptive_sampling,
                )
            ]
        ),
        seed=seed,
    )


def _solver_legacy_views(solver):
    state = solver.state
    data_output = xr.DataOutput(
        state.normalized_data,
        state.basal_mask,
        state.sub_region_indices,
        state.scales,
    )
    pred_u = state.solution[0]
    net = lambda params, x, idx: pred_u(params, x, idx)
    eqn_fn = lambda params, x, idx: djax.ssa_iso(
        lambda z: net(params, z, idx),
        x,
        state.scales[idx],
        basal=state.basal_mask[idx],
    )
    eval_f = lambda params, x, idx: jax.vmap(lambda z: eqn_fn(params, z, idx), in_axes=(0,))(x)
    xpinn_output = xr.XPINNOutput(state.params, state.solution, eval_f)
    return data_output, xpinn_output


def run_solver_kfac_experiment(
    iterations=1500,
    seed=8132002,
    tag=None,
    log_rate=100,
    depth=DEFAULT_DEPTH,
    width=DEFAULT_WIDTH,
    sample_count=DEFAULT_SAMPLE_COUNT,
    interface_points=DEFAULT_INTERFACE_POINTS,
    calving_front_points=DEFAULT_CALVING_FRONT_POINTS,
    interface_collocation=DEFAULT_INTERFACE_COLLOCATION,
    use_gpinn=False,
    target_c_rel_mae=None,
    continue_chunk=500,
    max_iterations=None,
    checkpoint_dir=None,
    resume_checkpoint=None,
    legacy_kfac_eval=False,
    adaptive_sampling=DEFAULT_ADAPTIVE_SAMPLING,
):
    if jax.default_backend().lower() == "metal":
        raise RuntimeError("KFAC is unavailable on the JAX Metal backend. Run with JAX_PLATFORM_NAME=cpu.")

    tag = tag or f"{iterations}_kfac"
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _artifact_dir(tag) / "checkpoints"
    checkpoint_path = None
    solver_ref = {}

    def checkpoint_callback(state):
        nonlocal checkpoint_path
        solver = solver_ref["solver"]
        data_output, xpinn_output = _solver_legacy_views(solver)
        mu_rel_mae, c_rel_mae = _inversion_mae(state["params"], data_output, xpinn_output)
        checkpoint_path = checkpoint_dir / f"KFAC_step_{state['step']}.pkl"
        _save_checkpoint(
            checkpoint_path,
            state["step"],
            state["params"],
            state["opt_state"],
            state["damping"],
            state["key"],
            state["history"],
            mu_rel_mae,
            c_rel_mae,
            dict(
                seed=seed,
                depth=depth,
                width=width,
                sample_count=sample_count,
                requested_interface_points=interface_points,
                interface_point_counts=state["interface_point_counts"],
                calving_front_points=calving_front_points,
                interface_collocation=interface_collocation,
                use_gpinn=use_gpinn,
                legacy_kfac_eval=legacy_kfac_eval,
                adaptive_sampling=adaptive_sampling,
                resume_checkpoint="" if resume_checkpoint is None else str(resume_checkpoint),
            ),
            x_col_mem=state["x_col_mem"],
            adapted=state["adapted"],
        )
        print(
            f"KFAC checkpoint step {state['step']}: C_REL_MAE={c_rel_mae:.6f} "
            f"MU_REL_MAE={mu_rel_mae:.6f} path={checkpoint_path}",
            flush=True,
        )
        return dict(checkpoint_path=checkpoint_path, mu_rel_mae=mu_rel_mae, c_rel_mae=c_rel_mae)

    optimizer_parameters = dict(
        preset="xpinn_joint_inversion_reference",
        iterations=iterations,
        log_rate=log_rate,
        legacy_kfac_eval=legacy_kfac_eval,
        interface_points=interface_points,
        interface_collocation=interface_collocation,
        max_iterations=max_iterations,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
        target_c_rel_mae=target_c_rel_mae,
        use_gpinn=use_gpinn,
        adaptive_sampling=adaptive_sampling,
        active_regions=xr.ACTIVE_REGIONS,
        eqn_weight_regions=xr.EQN_WEIGHT_REGIONS,
        checkpoint_callback=checkpoint_callback,
    )
    solver = _build_solver(
        seed,
        depth,
        width,
        sample_count,
        interface_collocation,
        use_gpinn,
        adaptive_sampling,
        optimizer_parameters,
    ).prepare()
    solver_ref["solver"] = solver
    state = solver.state
    optimizer_parameters["eval_f"] = lambda params, x, idx, basal: djax.ssa_iso(
        lambda z: state.solution[0](params, z, idx),
        x,
        state.scales[idx],
        basal=basal,
    )
    solver.fit()
    kfac_state = solver.kfac_state
    solver.state.params = kfac_state["params"]
    data_output, xpinn_output = _solver_legacy_views(solver)
    initial_objective = kfac_state["initial_objective"]
    final_objective = kfac_state["final_objective"]
    elapsed = kfac_state["elapsed"]
    start_step = kfac_state["start_step"]
    final_step = kfac_state["final_step"]
    history = kfac_state["history"]
    checkpoint_path = kfac_state["checkpoint_path"] or checkpoint_path

    output_path = _artifact_path(tag, ".npz")
    comparison_path = _artifact_path(tag, "_fields.png")
    loss_curve_path = _artifact_path(tag, "_loss.png")
    equation_residual_path = _artifact_path(tag, "_equation_residuals.png")
    x_term_ratio_path = _artifact_path(tag, "_x_term_ratios.png")
    mu_regions, c_regions = _collect_inversion_fields(kfac_state["params"], data_output, xpinn_output)
    mu_rel_mae, c_rel_mae = save_inversion_comparison(comparison_path, mu_regions, c_regions)
    residual_regions, term_regions = _collect_equation_diagnostics(kfac_state["params"], data_output, xpinn_output)
    save_equation_residuals(equation_residual_path, residual_regions)
    save_x_equation_term_ratios(x_term_ratio_path, term_regions, residual_regions["x"])
    save_loss_curve(loss_curve_path, history, target_c_rel_mae)
    np.savez(
        output_path,
        optimizer="KFAC",
        requested_iterations=iterations,
        start_iteration=start_step,
        final_iteration=final_step,
        trained_iterations=final_step - start_step,
        elapsed_seconds=elapsed,
        initial_objective=initial_objective,
        final_objective=final_objective,
        loss_history=np.asarray(history, dtype=float),
        data_path=str(DATA_PATH),
        sample_count=sample_count,
        requested_interface_points=np.nan if interface_points is None else interface_points,
        interface_point_counts=kfac_state["interface_point_counts"],
        calving_front_points=calving_front_points,
        interface_collocation=interface_collocation,
        use_gpinn=use_gpinn,
        legacy_kfac_eval=legacy_kfac_eval,
        adaptive_sampling=adaptive_sampling,
        depth=depth,
        width=width,
        mu_rel_mae=mu_rel_mae,
        c_rel_mae=c_rel_mae,
        target_c_rel_mae=np.nan if target_c_rel_mae is None else target_c_rel_mae,
        checkpoint_path="" if checkpoint_path is None else str(checkpoint_path),
        field_comparison_path=str(comparison_path),
        equation_residual_path=str(equation_residual_path),
        x_term_ratio_path=str(x_term_ratio_path),
        loss_curve_path=str(loss_curve_path),
    )
    return initial_objective, final_objective, elapsed, output_path, comparison_path, loss_curve_path, start_step, final_step, checkpoint_path, mu_rel_mae, c_rel_mae


def test_xpinn_joint_inversion_kfac_loss_decreases():
    initial_objective, final_objective, *_ = run_solver_kfac_experiment(iterations=3000, tag="3000_kfac")
    assert final_objective < initial_objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Synthetic XPINN joint-inversion KFAC smoke test.")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--log-rate", type=int, default=100)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--interface-points", type=int, default=DEFAULT_INTERFACE_POINTS)
    parser.add_argument("--calving-front-points", type=int, default=DEFAULT_CALVING_FRONT_POINTS)
    parser.add_argument("--interface-collocation", type=int, default=DEFAULT_INTERFACE_COLLOCATION)
    parser.add_argument("--use-gpinn", action="store_true")
    parser.add_argument("--target-c-rel-mae", type=float, default=None)
    parser.add_argument("--continue-chunk", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--plot-checkpoint", type=Path, default=None)
    parser.add_argument("--plot-cache", type=Path, default=None)
    parser.add_argument("--font-family", default=DEFAULT_FONT_FAMILY)
    parser.add_argument("--font-path", type=Path, default=DEFAULT_FONT_PATH)
    parser.add_argument("--legacy-kfac-eval", action="store_true")
    parser.add_argument("--legacy-runner", action="store_true")
    parser.add_argument(
        "--adaptive-sampling",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_SAMPLING,
    )
    args = parser.parse_args()
    configure_plot_font(args.font_family, args.font_path)
    if args.plot_cache is not None:
        output_path, comparison_path, loss_curve_path, equation_residual_path, x_term_ratio_path, plot_cache_path, mu_rel_mae, c_rel_mae = render_plot_cache(
            args.plot_cache,
            tag=args.tag,
            target_c_rel_mae=args.target_c_rel_mae,
        )
        print(f"KFAC_MU_REL_MAE={mu_rel_mae:.8f}")
        print(f"KFAC_C_REL_MAE={c_rel_mae:.8f}")
        print(f"KFAC_OUTPUT={output_path}")
        print(f"KFAC_FIELD_COMPARISON={comparison_path}")
        print(f"KFAC_EQUATION_RESIDUALS={equation_residual_path}")
        print(f"KFAC_X_TERM_RATIOS={x_term_ratio_path}")
        print(f"KFAC_LOSS_CURVE={loss_curve_path}")
        print(f"KFAC_PLOT_CACHE={plot_cache_path}")
        raise SystemExit(0)

    if args.plot_checkpoint is not None:
        output_path, comparison_path, loss_curve_path, equation_residual_path, x_term_ratio_path, plot_cache_path, mu_rel_mae, c_rel_mae = plot_checkpoint(
            args.plot_checkpoint,
            tag=args.tag,
            target_c_rel_mae=args.target_c_rel_mae,
        )
        print(f"KFAC_MU_REL_MAE={mu_rel_mae:.8f}")
        print(f"KFAC_C_REL_MAE={c_rel_mae:.8f}")
        print(f"KFAC_OUTPUT={output_path}")
        print(f"KFAC_FIELD_COMPARISON={comparison_path}")
        print(f"KFAC_EQUATION_RESIDUALS={equation_residual_path}")
        print(f"KFAC_X_TERM_RATIOS={x_term_ratio_path}")
        print(f"KFAC_LOSS_CURVE={loss_curve_path}")
        print(f"KFAC_PLOT_CACHE={plot_cache_path}")
        raise SystemExit(0)

    runner = run_kfac_experiment if args.legacy_runner else run_solver_kfac_experiment
    initial, final, elapsed, output_path, comparison_path, loss_curve_path, start_step, final_step, checkpoint_path, mu_rel_mae, c_rel_mae = runner(
        args.iterations,
        tag=args.tag,
        log_rate=args.log_rate,
        depth=args.depth,
        width=args.width,
        sample_count=args.sample_count,
        interface_points=args.interface_points,
        calving_front_points=args.calving_front_points,
        interface_collocation=args.interface_collocation,
        use_gpinn=args.use_gpinn,
        target_c_rel_mae=args.target_c_rel_mae,
        continue_chunk=args.continue_chunk,
        max_iterations=args.max_iterations,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
        legacy_kfac_eval=args.legacy_kfac_eval,
        adaptive_sampling=args.adaptive_sampling,
    )
    print(f"KFAC_INITIAL_OBJECTIVE={initial:.8e}")
    print(f"KFAC_FINAL_OBJECTIVE={final:.8e}")
    print(f"KFAC_START_ITERATION={start_step}")
    print(f"KFAC_FINAL_ITERATION={final_step}")
    print(f"KFAC_SECONDS={elapsed:.6f}")
    trained_iterations = final_step - start_step
    seconds_per_iter = elapsed / trained_iterations if trained_iterations else np.nan
    print(f"KFAC_SECONDS_PER_ITER={seconds_per_iter:.8f}")
    print(f"KFAC_MU_REL_MAE={mu_rel_mae:.8f}")
    print(f"KFAC_C_REL_MAE={c_rel_mae:.8f}")
    print(f"KFAC_CHECKPOINT={checkpoint_path}")
    print(f"KFAC_OUTPUT={output_path}")
    print(f"KFAC_FIELD_COMPARISON={comparison_path}")
    print(f"KFAC_LOSS_CURVE={loss_curve_path}")
