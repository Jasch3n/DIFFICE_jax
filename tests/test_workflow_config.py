import json
import pickle

import jax.numpy as jnp

import diffice_jax as djax
import diffice_jax.core.solver as solver_module
from diffice_jax.core.solver import PreparedState
from diffice_jax.data.xpinns import sampling as xpinn_sampling


def _pinn_config():
    return """
name: small_pinn
workflow: ice-shelf-only
data:
  source: data_pinns_test.mat
  sampling_counts:
    velocity_data: 4
    thickness_data: 5
    surface_data: null
    collocation: 6
    calving_front: 7
  collocation_library_size: full
model:
  workflow: pinn
  network:
    depth: 2
    width: 8
equation:
  name: ssa_iso
loss:
  name: iso
  weights: [1.0, 0.05, 0.1]
training:
  seed: 11
  stages:
    - optimizer:
        name: adam
        learning_rate: 0.001
      iterations: 3
artifacts:
  output_dir: outputs/{tag}
"""


def _xpinn_config_dict():
    return {
        "name": "small_xpinn",
        "workflow": "joint-inversion-regression",
        "data": {
            "source": "flatbed.mat",
            "interface_collocation": {
                "library_size": 9,
                "sample_count": 5,
            },
            "sampling_counts": {
                "velocity_data": [4, 4],
                "thickness_data": [4, 4],
                "surface_data": [4, 4],
                "collocation": [4, 4],
            },
        },
        "model": {
            "workflow": "xpinn",
            "regions": [
                {"index": 0, "kind": "grounded"},
                {"index": 1, "kind": "floating"},
            ],
            "network": {"depth": 2, "width": 8},
        },
        "equation": {"name": "ssa_iso"},
        "loss": {
            "name": "joint_inversion_regression",
            "matching": True,
            "calving_front": True,
            "use_gpinn": True,
            "active_regions": [0, 1],
        },
        "training": {
            "seed": 13,
            "global_weights": {
                "data": 1.0,
                "equation": 0.01,
                "calving_front": 0.01,
                "matching": 0.5,
                "gpinn": 0.001,
                "mu_gradient": 0.0,
            },
            "stages": [
                {
                    "optimizer": {
                        "name": "kfac",
                        "learning_rate": None,
                        "damping": "nan",
                        "parameters": {
                            "checkpoint_dir": "checkpoints",
                            "interface_points": "all",
                        },
                    },
                    "iterations": 2,
                    "adaptive_sampling": True,
                    "adaptive_sampling_burn_in": 37,
                    "adaptive_sampling_period": 11,
                }
            ],
        },
    }


def test_yaml_workflow_config_builds_pinn_solver(tmp_path):
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(_pinn_config(), encoding="utf-8")

    config = djax.load_workflow_config(config_path)
    solver = djax.build_solver_from_config(config)

    assert solver.seed == 11
    assert solver.model_config.workflow == "pinn"
    assert solver.data_config.source == tmp_path / "data_pinns_test.mat"
    assert jnp.array_equal(solver.data_config.sampling_counts, jnp.array([4, 5, 6, 7], dtype="int32"))
    assert solver.training_config.stages[0].optimizer.name == "adam"


def test_legacy_json_workflow_config_builds_xpinn_solver(tmp_path):
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(_xpinn_config_dict()), encoding="utf-8")

    config = djax.load_workflow_config(config_path)
    solver = djax.build_solver_from_config(config)
    stage = solver.training_config.stages[0]

    assert solver.seed == 13
    assert solver.data_config.regression_workflow is True
    assert solver.loss_config.name == "joint_inversion_regression"
    assert solver.loss_config.use_gpinn is True
    assert solver.loss_config.global_weights["matching"] == 0.5
    assert solver.loss_config.global_weights["gpinn"] == 0.001
    assert solver.model_config.regions == [
        djax.RegionConfig("grounded", 0),
        djax.RegionConfig("floating", 1),
    ]
    assert stage.optimizer.name == "kfac"
    assert jnp.isnan(stage.optimizer.damping)
    assert stage.optimizer.parameters["checkpoint_dir"] == tmp_path / "checkpoints"
    assert stage.optimizer.parameters["interface_points"] is None
    assert stage.adaptive_sampling is True
    assert stage.adaptive_sampling_burn_in == 37
    assert stage.adaptive_sampling_period == 11
    assert xpinn_sampling.N_INTERFACE_LIBRARY == 9
    assert xpinn_sampling.N_INTERFACE_COLLOCATION == 5
    assert solver.data_config.sampling_counts == [[4, 4], [4, 4], [4, 4]]


def test_adaptive_sampling_burnin_alias_is_normalized(tmp_path):
    raw = _xpinn_config_dict()
    raw["training"]["stages"][0].pop("adaptive_sampling_burn_in")
    raw["training"]["stages"][0]["adaptive_sampling_burnin"] = 41
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    stage = djax.build_solver_from_config(djax.load_workflow_config(config_path)).training_config.stages[0]

    assert stage.adaptive_sampling_burn_in == 41
    assert stage.adaptive_sampling_period == 11


def test_adaptive_sampling_under_optimizer_parameters_is_rejected(tmp_path):
    raw = _xpinn_config_dict()
    raw["training"]["stages"][0]["optimizer"]["parameters"]["adaptive_sampling_period"] = 10
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        djax.build_solver_from_config(djax.load_workflow_config(config_path))
    except ValueError as exc:
        assert "Use training.stages[].adaptive_sampling_period instead" in str(exc)
    else:
        raise AssertionError("Expected optimizer.parameters adaptive sampling controls to fail.")


def test_use_gpinn_under_optimizer_parameters_is_rejected(tmp_path):
    raw = _xpinn_config_dict()
    raw["loss"].pop("use_gpinn", None)
    raw["training"]["stages"][0]["optimizer"]["parameters"]["use_gpinn"] = True
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        djax.build_solver_from_config(djax.load_workflow_config(config_path))
    except ValueError as exc:
        assert "Use loss.use_gpinn instead" in str(exc)
    else:
        raise AssertionError("Expected optimizer.parameters.use_gpinn to fail.")


def test_per_region_network_aliases_are_normalized(tmp_path):
    raw = _xpinn_config_dict()
    raw["model"]["per_region_networks"] = [
        {"u": [3, 9], "mu": [2, 8]},
        {"u": [4, 10], "mu": [1, 7], "c0": [2, 6]},
    ]
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    solver = djax.build_solver_from_config(djax.load_workflow_config(config_path))

    assert solver.model_config.per_region_networks == [
        {"net_u": {"depth": 3, "width": 9}, "net_mu": {"depth": 2, "width": 8}},
        {
            "net_u": {"depth": 4, "width": 10},
            "net_mu": {"depth": 1, "width": 7},
            "net_c": {"depth": 2, "width": 6},
        },
    ]


def test_positional_sampling_counts_still_build_pinn_solver(tmp_path):
    config_path = tmp_path / "workflow.json"
    raw = {
        "name": "legacy_pinn",
        "data": {
            "source": "data_pinns_test.mat",
            "sampling_counts": [4, 5, 6, 7],
        },
        "model": {
            "workflow": "pinn",
            "network": {"depth": 2, "width": 8},
        },
        "equation": {"name": "ssa_iso"},
        "loss": {"name": "iso"},
        "training": {"seed": 17, "stages": []},
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    solver = djax.build_solver_from_config(djax.load_workflow_config(config_path))

    assert jnp.array_equal(solver.data_config.sampling_counts, jnp.array([4, 5, 6, 7], dtype="int32"))


def test_public_workflow_validates_model_family(tmp_path):
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        """
name: mismatch
workflow: joint-inversion
data:
  source: data_pinns_test.mat
  sampling_counts: [4, 5, 6, 7]
model:
  workflow: pinn
  network:
    depth: 2
    width: 8
training:
  stages: []
""",
        encoding="utf-8",
    )

    config = djax.load_workflow_config(config_path)

    try:
        djax.build_solver_from_config(config)
    except ValueError as exc:
        assert "expects model.workflow='xpinn'" in str(exc)
    else:
        raise AssertionError("Expected workflow/model mismatch to fail.")


def test_joint_inversion_workflow_uses_non_regression_sampler(tmp_path):
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        """
name: joint
workflow: joint_inversion
data:
  source: flatbed.mat
  sampling_counts:
    velocity_data: [4, 4]
    thickness_data: [4, 4]
    surface_data: [4, 4]
    collocation: [4, 4]
    calving_front: [2, 2]
    matching: 3
model:
  workflow: xpinn
  regions:
    - index: 0
      kind: grounded
    - index: 1
      kind: floating
  network:
    depth: 2
    width: 8
loss:
  name: joint_inversion
training:
  stages: []
""",
        encoding="utf-8",
    )

    solver = djax.build_solver_from_config(djax.load_workflow_config(config_path))

    assert solver.data_config.regression_workflow is False
    assert solver.loss_config.name == "joint_inversion"
    assert solver.data_config.sampling_counts == [[4, 4], [4, 4], [4, 4], [2, 2], 3]


def test_solver_save_serializes_workflow_paths_and_arrays(tmp_path):
    solver = djax.DIFFICESolver(
        data=djax.DataConfig(
            source=tmp_path / "data.mat",
            sampling_counts=jnp.array([1, 2, 3, 4], dtype="int32"),
        ),
        model=djax.ModelConfig(workflow="pinn", network=djax.NetworkConfig(depth=2, width=8)),
        training=djax.TrainingConfig(
            stages=[
                djax.TrainingStage(
                    optimizer=djax.OptimizerConfig(
                        name="kfac",
                        damping=jnp.nan,
                        parameters={"checkpoint_dir": tmp_path / "checkpoints"},
                    ),
                    iterations=1,
                )
            ]
        ),
        seed=5,
    )
    solver.state = PreparedState(
        raw_data=None,
        normalized_data=None,
        sub_region_indices=None,
        position_data=None,
        crop_indices=None,
        basal_mask=[False],
        scales=None,
        params={"w": jnp.array([1.0])},
        solution=None,
        dataf=None,
        lossf=None,
        initial_loss=jnp.array(0.0),
    )
    predictions = {
        "workflow": "pinn",
        "regions": [
            {
                "u": jnp.ones((2, 2)),
                "v": jnp.ones((2, 2)) * 2.0,
                "h": jnp.ones((2, 2)) * 3.0,
                "mu": jnp.ones((2, 2)) * 4.0,
            }
        ],
    }
    solver.predict = lambda: predictions

    solver.save(tmp_path / "solver")

    with open(tmp_path / "solver" / "config.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["data"]["source"] == str(tmp_path / "data.mat")
    assert saved["data"]["sampling_counts"] == [1, 2, 3, 4]
    assert saved["training"]["stages"][0]["optimizer"]["parameters"]["checkpoint_dir"] == str(tmp_path / "checkpoints")
    with open(tmp_path / "solver" / "predictions.pkl", "rb") as f:
        predictions = pickle.load(f)
    assert predictions["workflow"] == "pinn"
    assert predictions["regions"][0]["u"].shape == (2, 2)


def test_solver_load_params_accepts_plain_params_and_checkpoint_dict(tmp_path):
    solver = djax.DIFFICESolver(
        data=djax.DataConfig(source=tmp_path / "data.mat", sampling_counts=jnp.array([1, 2, 3, 4])),
        model=djax.ModelConfig(workflow="pinn", network=djax.NetworkConfig(depth=2, width=8)),
    )
    solver.state = PreparedState(
        raw_data=None,
        normalized_data=None,
        sub_region_indices=None,
        position_data=None,
        crop_indices=None,
        basal_mask=[False],
        scales=None,
        params={"w": jnp.array([0.0])},
        solution=None,
        dataf=None,
        lossf=None,
        initial_loss=jnp.array(0.0),
    )

    plain_path = tmp_path / "params.pkl"
    with open(plain_path, "wb") as f:
        pickle.dump({"w": jnp.array([1.0])}, f)
    solver.load_params(plain_path)
    assert float(solver.state.params["w"][0]) == 1.0

    checkpoint_path = tmp_path / "checkpoint.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"step": 3, "params": {"w": jnp.array([2.0])}}, f)
    solver.load_params(checkpoint_path)
    assert float(solver.state.params["w"][0]) == 2.0


def test_solver_adaptive_eval_vectorizes_pointwise_equation(monkeypatch, tmp_path):
    def fake_gov_eqn(net, x, scale, basal=False):
        if x.ndim != 1:
            raise AssertionError("RAD evaluator must call gov_eqn_iso on one collocation point.")
        n_terms = 9 if basal else 7
        return jnp.array([x[0], x[1]]), jnp.ones((n_terms,)) * scale

    monkeypatch.setattr(solver_module, "gov_eqn_iso", fake_gov_eqn)
    solver = djax.DIFFICESolver(
        data=djax.DataConfig(source=tmp_path / "data.mat", sampling_counts=[[1], [1], [1]]),
        model=djax.ModelConfig(
            workflow="xpinn",
            regions=[
                djax.RegionConfig("grounded", 5),
                djax.RegionConfig("floating", 7),
            ],
        ),
    )
    solver.state = PreparedState(
        raw_data=None,
        normalized_data=None,
        sub_region_indices=[5, 7],
        position_data=None,
        crop_indices=None,
        basal_mask=[True, False],
        scales=[5.0, 7.0],
        params={},
        solution=(lambda params, x, idx: x,),
        dataf=None,
        lossf=None,
        initial_loss=jnp.array(0.0),
    )

    residual, terms = solver._adaptive_eval({})({}, jnp.ones((3, 2)), 7, False)

    assert residual.shape == (3, 2)
    assert terms.shape == (3, 7)
    assert jnp.all(terms == 7.0)
