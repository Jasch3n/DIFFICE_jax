import sys
import types
from pathlib import Path

import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'test_xpinn_regression'
sys.path.insert(0, str(SCRIPT_DIR))

if 'kfac_jax' not in sys.modules:
    kfac_stub = types.ModuleType('kfac_jax')
    kfac_stub.curvature_estimator = types.SimpleNamespace()
    kfac_stub.curvature_blocks = types.SimpleNamespace()
    kfac_stub.loss_functions = types.SimpleNamespace()
    kfac_stub.optimizer = types.SimpleNamespace()
    sys.modules['kfac_jax'] = kfac_stub

import calc_char
import xpinn_regression


def test_pq_rg_terms_match_characteristic_reduction():
    u = 4.0
    v = 3.0
    h = 10.0
    ux = 2.0
    uy = 5.0
    vx = 7.0
    vy = 11.0
    sx = 13.0
    sy = 17.0
    div1 = 19.0
    div2 = 23.0

    p, q, exx2, exy, eyy2 = calc_char.pq_terms_from_kinematics(u, v, ux, uy, vx, vy)
    assert np.isclose(exx2, 4.0 * ux + 2.0 * vy)
    assert np.isclose(exy, uy + vx)
    assert np.isclose(eyy2, 4.0 * vy + 2.0 * ux)
    assert np.isclose(p, v * exx2 - u * exy)
    assert np.isclose(q, v * exy - u * eyy2)

    p2, q2, r, g = calc_char.rg_terms_from_kinematics(u, v, h, sx, sy, ux, uy, vx, vy, div1, div2)
    assert np.isclose(p2, p)
    assert np.isclose(q2, q)
    assert np.isclose(r, v * div1 - u * div2)
    assert np.isclose(g, calc_char.RHO_I * calc_char.G * h * (v * sx - u * sy))


def test_trace_characteristic_advects_and_integrates_hnu():
    def local_eval(_xy):
        return {
            'p': 2.0,
            'q': 0.0,
            'r': 0.0,
            'g': 4.0,
            'u': 1.0,
            'v': 1.0,
            'h': 2.0,
        }

    curve = calc_char.trace_characteristic(
        seed_xy=np.array([0.0, 0.0]),
        seed_mu=3.0,
        local_eval=local_eval,
        bounds=np.array([[-1.0, 10.0], [-1.0, 1.0]]),
        s_max=2.0,
        ds=1.0,
        unit_speed=False,
    )

    assert len(curve) == 3
    assert np.isclose(curve[0]['hnu'], 6.0)
    assert np.isclose(curve[1]['x'], 2.0)
    assert np.isclose(curve[2]['x'], 4.0)
    assert np.isclose(curve[2]['hnu'], 14.0)


def test_extract_grounded_region_uses_processed_xpinn_layout():
    data_output = xpinn_regression.load_data(str(xpinn_regression.DATA_PATH))
    region = calc_char.extract_grounded_region(data_output)

    assert data_output.basal_mask[region['region_idx']] is True
    assert region['vel_xy'].shape[1] == 2
    assert region['thk_xy'].shape[1] == 2
    assert region['seed_xy'].shape[1] == 2
    assert region['seed_mu'].shape[1] == 1
    assert np.isfinite(region['seed_xy']).all()
    assert np.isfinite(region['seed_mu']).all()


def test_run_characteristics_smoke(tmp_path):
    data_output = xpinn_regression.load_data(str(xpinn_regression.DATA_PATH))
    result = calc_char.run_characteristics(
        data_output,
        random.PRNGKey(0),
        steps=5,
        seed_count=3,
        s_max=4.0e3,
        ds=2.0e3,
        width=8,
        depth=1,
    )

    assert result['flat_curves']['x'].size > 0
    assert result['hnu_grid'].shape == result['grounded']['x_grid'].shape
    assert result['beta_grid'].shape == result['grounded']['x_grid'].shape

    output = tmp_path / 'calc_char_smoke.npz'
    class Args:
        steps = 5
        lr = calc_char.LEARNING_RATE
        seed_count = 3
        s_max = 4.0e3
        ds = 2.0e3

    calc_char.save_results(output, result, Args)
    saved = np.load(output)
    assert saved['x'].size > 0
