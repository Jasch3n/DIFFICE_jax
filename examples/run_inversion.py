from pathlib import Path
import argparse
import json
import importlib.util

import numpy as np


def _runtime_env_helper():
    helper_path = Path(__file__).resolve().parents[1] / "diffice_jax" / "workflow" / "runtime_env.py"
    spec = importlib.util.spec_from_file_location("_diffice_runtime_env", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_module():
    module_path = Path(__file__).resolve().parent / "render_solver_xpinn_kfac_plots.py"
    spec = importlib.util.spec_from_file_location("_diffice_render_plots", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maybe_render_plots(config, result, args):
    """Render XPINN plots into <output_dir>/plots after a saved run.

    On by default; skipped for --no-plot, --no-save (no output dir), and
    non-XPINN workflows. A plotting failure never fails a completed run.
    """
    if args.no_plot:
        return
    if args.no_save or result.output_dir is None:
        print("PLOTTING_SKIPPED=no-output-dir")
        return
    if config.model.get("workflow") != "xpinn":
        print(f"PLOTTING_SKIPPED=workflow-{config.model.get('workflow')}")
        return
    tag = str(config.artifacts.get("tag", config.name))
    try:
        render = _render_module()
        output, fields, data_fields, loss, residuals, ratios, cache, mu_rel_mae, c_rel_mae = (
            render.render_from_solver(result.solver, config, result.output_dir, tag)
        )
        print(f"PLOT_OUTPUT={output}")
        print(f"FIELD_COMPARISON={fields}")
        print(f"DATA_FIELD_COMPARISON={data_fields}")
        print(f"LOSS_CURVE={loss}")
        print(f"EQUATION_RESIDUALS={residuals}")
        print(f"X_TERM_RATIOS={ratios}")
        print(f"PLOT_CACHE={cache}")
        print(f"MU_REL_MAE={mu_rel_mae:.6f}")
        print(f"C_REL_MAE={c_rel_mae:.6f}")
    except Exception as exc:  # plots must never fail an already-completed run
        import traceback

        print(f"PLOTTING_FAILED={type(exc).__name__}: {exc}")
        traceback.print_exc()


def _preload_runtime(path):
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        return
    runtime = raw.get("runtime", {})
    helper = _runtime_env_helper()
    env = helper.apply_runtime_env(runtime, emit=False)
    print(f"JAX_CACHE_CONFIG={json.dumps(env, sort_keys=True)}", flush=True)


def _count_points(value, n_regions):
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(int(item) for item in value)
    return int(value) * n_regions


def _resolve_config_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else config.base_dir / path


def _field_arrays(data, key):
    if key not in data:
        return []
    value = data[key]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [np.asarray(item).reshape(-1) for item in value.reshape(-1)]
    return [np.asarray(value).reshape(-1)]


def _count_finite_field(data, *keys):
    arrays = [_field_arrays(data, key) for key in keys]
    if not arrays or any(len(items) == 0 for items in arrays):
        return 0
    total = 0
    for items in zip(*arrays):
        mask = np.ones(items[0].shape, dtype=bool)
        for item in items:
            mask &= np.isfinite(item)
        total += int(mask.sum())
    return total


def _dataset_point_counts(config):
    from scipy.io import loadmat

    data = loadmat(str(_resolve_config_path(config, config.data["source"])))
    velocity = _count_finite_field(data, "ud", "vd")
    thickness = _count_finite_field(data, "hd")
    surface = _count_finite_field(data, "sd")
    collocation = _count_finite_field(data, "xcol", "ycol")
    interface = _count_finite_field(data, "x_md", "y_md")
    if collocation == 0:
        collocation = _count_finite_field(data, "xd", "yd", "ud", "vd")
    return velocity, thickness, surface, collocation, interface


def _interface_collocation_regions(config):
    regions = sorted(config.model.get("regions", []), key=lambda region: int(region.get("index", 0)))
    if not regions:
        return 0
    kinds = [region.get("kind", region.get("region_kind", "floating")) for region in regions]
    active = set()
    for pos in range(len(kinds) - 1):
        if kinds[pos] != kinds[pos + 1]:
            active.add(pos)
            active.add(pos + 1)
    return len(active)


def _print_point_counts(config):
    velocity_total, thickness_total, surface_total, collocation_total, interface_total = _dataset_point_counts(config)
    print(f"DATASET_VELOCITY_DATA_POINTS={velocity_total}", flush=True)
    print(f"DATASET_THICKNESS_DATA_POINTS={thickness_total}", flush=True)
    print(f"DATASET_SURFACE_DATA_POINTS={surface_total}", flush=True)
    print(f"DATASET_COLLOCATION_POINTS={collocation_total}", flush=True)
    print(f"DATASET_INTERFACE_POINTS={interface_total}", flush=True)

    sampling = config.data.get("sampling_counts", {})
    if not isinstance(sampling, dict):
        return
    n_regions = max(len(config.model.get("regions", [])), 1)
    velocity = _count_points(sampling.get("velocity_data"), n_regions)
    thickness = _count_points(sampling.get("thickness_data"), n_regions)
    surface = _count_points(sampling.get("surface_data", sampling.get("thickness_data")), n_regions)
    regular_collocation = _count_points(sampling.get("collocation"), n_regions)
    interface = _count_points(sampling.get("matching"), max(n_regions - 1, 1))
    interface_cfg = config.data.get("interface_collocation", {})
    interface_collocation = (
        int(interface_cfg.get("sample_count", 0)) * _interface_collocation_regions(config)
        if isinstance(interface_cfg, dict) else 0
    )
    total_collocation = regular_collocation + interface_collocation
    print(f"BATCH_VELOCITY_DATA_POINTS={velocity}", flush=True)
    print(f"BATCH_THICKNESS_DATA_POINTS={thickness}", flush=True)
    print(f"BATCH_SURFACE_DATA_POINTS={surface}", flush=True)
    print(f"BATCH_INTERFACE_POINTS={interface}", flush=True)
    print(f"BATCH_REGULAR_COLLOCATION_POINTS={regular_collocation}", flush=True)
    print(f"BATCH_INTERFACE_COLLOCATION_POINTS={interface_collocation}", flush=True)
    print(f"BATCH_COLLOCATION_POINTS={total_collocation}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run a DIFFICE_jax training workflow from config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip rendering XPINN plots after a saved run.")
    args = parser.parse_args()

    _preload_runtime(args.config)
    from diffice_jax.workflow import load_workflow_config, run_training_workflow

    config = load_workflow_config(args.config)
    _print_point_counts(config)
    result = run_training_workflow(config, save=not args.no_save)
    print(f"WORKFLOW_NAME={config.name}")
    print(f"WORKFLOW_SECONDS={result.elapsed_seconds:.6f}")
    kfac_state = getattr(result.solver, "kfac_state", None)
    if kfac_state is not None:
        trained_iterations = int(kfac_state["final_step"]) - int(kfac_state["start_step"])
        seconds_per_iter = float(kfac_state.get("seconds_per_iter", float("nan")))
        print(f"WORKFLOW_START_ITERATION={int(kfac_state['start_step'])}")
        print(f"WORKFLOW_FINAL_ITERATION={int(kfac_state['final_step'])}")
        print(f"WORKFLOW_TRAINED_ITERATIONS={trained_iterations}")
        print(f"WORKFLOW_TIMED_ITERATIONS={int(kfac_state.get('timed_iterations', 0))}")
        print(f"WORKFLOW_TIMED_SECONDS={float(kfac_state.get('timed_elapsed', 0.0)):.6f}")
        print(f"WORKFLOW_SECONDS_PER_ITER={seconds_per_iter:.8f}")
    if result.output_dir is not None:
        print(f"WORKFLOW_OUTPUT_DIR={result.output_dir}")

    _maybe_render_plots(config, result, args)


if __name__ == "__main__":
    main()
