from pathlib import Path
import argparse
import time

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.io import loadmat

import diffice_jax as djax

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_DIR = Path(__file__).parent / "figures"
ARTIFACT_PREFIX = "test_pinn_synthetic_ice_shelf"


def artifact_dir(tag):
    path = FIGURE_DIR / f"ice_shelf_only__{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(tag, suffix):
    return artifact_dir(tag) / f"{ARTIFACT_PREFIX}_{tag}{suffix}"


def build_solver(raw):
    """Build the standalone PINN solver used for synthetic ice-shelf inversion."""

    return djax.DIFFICESolver(
        data=djax.DataConfig(
            source=raw,
            sampling_counts=jnp.array([1024, 1024, 2048, 256], dtype="int32"),
            collocation_library_size="full",
        ),
        model=djax.ModelConfig(
            workflow="pinn",
            network=djax.NetworkConfig(depth=6, width=30),
        ),
        equation=djax.EquationConfig(name="ssa_iso"),
        loss=djax.LossConfig(name="iso", weights=(1.0, 0.05, 0.1)),
        training=djax.TrainingConfig(stages=[]),
        seed=1234,
    )


def viscosity_fields_and_relative_mae(solver, mu_true_full):
    """Return gridded viscosity fields and MAE normalized by mean truth magnitude."""

    idxval = np.asarray(solver.state.normalized_data[4][4][0])
    dsize = solver.state.normalized_data[4][5][0]
    mu0 = float(solver.state.scales[0].dynamic_scale.mu0)
    mu_true_flat = mu_true_full.reshape(-1)
    mu_pred_valid = np.asarray(solver.predict()["named"]["mu"]).reshape(-1) * mu0
    mu_true_valid = mu_true_flat[idxval]
    rel_mae = float(np.mean(np.abs(mu_pred_valid - mu_true_valid)) / np.mean(np.abs(mu_true_valid)))

    mu_pred = np.full(mu_true_flat.shape, np.nan)
    mu_pred[idxval] = mu_pred_valid
    mu_pred = mu_pred.reshape(dsize)
    mu_true = mu_true_full.reshape(dsize)
    mismatch = np.abs(mu_pred - mu_true) / np.maximum(np.abs(mu_true), 1e-12)
    return mu_true, mu_pred, mismatch, rel_mae


def save_viscosity_comparison(path, mu_true, mu_pred, mismatch, rel_mae):
    """Save a three-panel spatial comparison for true, inferred, and mismatch fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    vmin = np.nanpercentile(mu_true, 2)
    vmax = np.nanpercentile(mu_true, 98)
    mismatch_vmax = max(0.5, np.nanpercentile(mismatch, 98))

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (mu_true, "True effective viscosity", "viridis", vmin, vmax, "Pa s"),
        (mu_pred, "Inferred effective viscosity", "viridis", vmin, vmax, "Pa s"),
        (mismatch, f"Relative absolute mismatch\nMAE = {100 * rel_mae:.2f}%", "magma", 0.0, mismatch_vmax, "|error| / |truth|"),
    ]
    for ax, (field, title, cmap, lo, hi, label) in zip(axs, panels):
        image = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        cbar = fig.colorbar(image, ax=ax, shrink=0.88)
        cbar.set_label(label)
    fig.suptitle("Synthetic ice-shelf PINN viscosity inference", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_run_artifacts(path, optimizer, iterations, elapsed_seconds, rel_mae, loss_history):
    """Persist training diagnostics for optimizer-to-optimizer comparisons."""

    path.parent.mkdir(parents=True, exist_ok=True)
    loss_history = np.asarray(loss_history, dtype=float)
    actual_iterations = loss_history.shape[0]
    np.savez(
        path,
        optimizer=optimizer,
        requested_iterations=iterations,
        iterations=actual_iterations,
        elapsed_seconds=elapsed_seconds,
        seconds_per_iteration=elapsed_seconds / actual_iterations,
        rel_mae=rel_mae,
        loss_history=loss_history,
    )


def run_adam_experiment(iterations=5000, tag=None):
    """Run Adam on the synthetic ice-shelf fixture and save figures/results."""

    tag = tag or f"{iterations}_adam"
    raw = loadmat(Path(__file__).with_name("data_pinns_test.mat"))
    solver = build_solver(raw)
    solver.prepare()
    solver.training_config = djax.TrainingConfig(
        stages=[
            djax.TrainingStage(
                djax.OptimizerConfig("adam", learning_rate=1e-3),
                iterations=iterations,
            )
        ]
    )

    start = time.perf_counter()
    solver.fit()
    elapsed = time.perf_counter() - start

    mu_true, mu_pred, mismatch, rel_mae = viscosity_fields_and_relative_mae(solver, raw["mud"])
    save_viscosity_comparison(
        artifact_path(tag, ".png"),
        mu_true,
        mu_pred,
        mismatch,
        rel_mae,
    )
    save_run_artifacts(
        artifact_path(tag, ".npz"),
        "Adam",
        iterations,
        elapsed,
        rel_mae,
        solver.loss_history,
    )
    return solver, rel_mae, elapsed


def save_optimizer_comparison(adam_path, kfac_path, output_path):
    """Plot PINN loss curves against iteration and wall-clock training time."""

    adam = np.load(adam_path)
    kfac = np.load(kfac_path)
    runs = [adam, kfac]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    for run in runs:
        label = (
            f"{run['optimizer'].item()} "
            f"({float(run['seconds_per_iteration']):.3f} s/iter, "
            f"MAE {100 * float(run['rel_mae']):.2f}%)"
        )
        loss = run["loss_history"][:, 0]
        steps = np.arange(1, loss.shape[0] + 1)
        seconds = steps * float(run["seconds_per_iteration"])
        axs[0].semilogy(steps, loss, label=label)
        axs[1].semilogy(seconds, loss, label=label)

    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Normalized loss")
    axs[0].set_title("Loss vs iteration")
    axs[1].set_xlabel("Training time (s)")
    axs[1].set_ylabel("Normalized loss")
    axs[1].set_title("Loss vs wall-clock time")
    for ax in axs:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic ice-shelf PINN optimizer comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def test_pinn_synthetic_ice_shelf_viscosity_inference():
    """Train the new solver interface on the legacy synthetic ice-shelf PINN fixture."""

    raw = loadmat(Path(__file__).with_name("data_pinns_test.mat"))
    solver, rel_mae, _ = run_adam_experiment(iterations=5000, tag="5000_adam")
    mu_true, mu_pred, mismatch, rel_mae = viscosity_fields_and_relative_mae(solver, raw["mud"])
    save_viscosity_comparison(
        artifact_path("baseline", ".png"),
        mu_true,
        mu_pred,
        mismatch,
        rel_mae,
    )
    assert rel_mae <= 0.05


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic ice-shelf Adam benchmark.")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--compare-kfac", default=None)
    args = parser.parse_args()
    tag = args.tag or f"{args.iterations}_adam"
    _, rel_mae, elapsed = run_adam_experiment(iterations=args.iterations, tag=tag)
    result = np.load(artifact_path(tag, ".npz"))
    print(f"ADAM_REL_MAE={rel_mae:.8f}")
    print(f"ADAM_SECONDS={elapsed:.6f}")
    print(f"ADAM_ACTUAL_ITERATIONS={int(result['iterations'])}")
    print(f"ADAM_SECONDS_PER_ITER={float(result['seconds_per_iteration']):.8f}")
    if args.compare_kfac is not None:
        save_optimizer_comparison(
            artifact_path(tag, ".npz"),
            Path(args.compare_kfac),
            artifact_path("optimizer_comparison", ".png"),
        )
