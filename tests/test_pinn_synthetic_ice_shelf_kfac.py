from pathlib import Path
import argparse
import time

import jax.numpy as jnp
from scipy.io import loadmat

import diffice_jax as djax
from tests.test_pinn_synthetic_ice_shelf import (
    artifact_path,
    build_solver,
    save_run_artifacts,
    save_viscosity_comparison,
    viscosity_fields_and_relative_mae,
)


ADAM_2500_REL_MAE = 0.12405031


def kfac_config():
    """Return the KFAC settings used for the synthetic ice-shelf PINN check."""

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
        inverse_update_period=1,
        num_burnin_steps=0,
        always_use_exact_qmodel_for_damping_adjustment=True,
        include_norms_in_stats=True,
    )


def run_kfac_experiment(iterations=1000, tag=None):
    """Run KFAC on the synthetic ice-shelf fixture and save figures/results."""

    tag = tag or f"{iterations}_kfac"
    raw = loadmat(Path(__file__).with_name("data_pinns_test.mat"))
    solver = build_solver(raw)
    solver.prepare()
    solver.training_config = djax.TrainingConfig(
        stages=[
            djax.TrainingStage(
                djax.OptimizerConfig("kfac", learning_rate=None, parameters=kfac_config()),
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
        "KFAC",
        iterations,
        elapsed,
        rel_mae,
        solver.loss_history,
    )
    return solver, rel_mae, elapsed


def test_pinn_synthetic_ice_shelf_kfac_viscosity_inference():
    """Train the synthetic ice-shelf PINN with KFAC and compare against Adam."""

    raw = loadmat(Path(__file__).with_name("data_pinns_test.mat"))
    solver, _, _ = run_kfac_experiment(iterations=1000, tag="1000_kfac")
    mu_true, mu_pred, mismatch, rel_mae = viscosity_fields_and_relative_mae(solver, raw["mud"])
    save_viscosity_comparison(
        artifact_path("1000_kfac", ".png"),
        mu_true,
        mu_pred,
        mismatch,
        rel_mae,
    )
    assert rel_mae <= ADAM_2500_REL_MAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic ice-shelf KFAC benchmark.")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    tag = args.tag or f"{args.iterations}_kfac"
    _, rel_mae, elapsed = run_kfac_experiment(iterations=args.iterations, tag=tag)
    print(f"KFAC_REL_MAE={rel_mae:.8f}")
    print(f"KFAC_SECONDS={elapsed:.6f}")
    print(f"KFAC_SECONDS_PER_ITER={elapsed / args.iterations:.8f}")
