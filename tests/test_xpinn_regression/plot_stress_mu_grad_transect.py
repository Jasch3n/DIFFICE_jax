import argparse
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).parent))
from xpinn_regression import initialize_xpinn, load_data

SCRIPT_DIR = Path(__file__).parent
TEST_NAME = 'flatbed'
CHECKPOINT_FOLDER = f'match_ct_eqn_{TEST_NAME}_checkpoints'
# CHECKPOINT_FOLDER = 'grounded_region0_from_floating_match_eqn_flatbed_checkpoints'
LOAD_STEP = 50000
CHECKPOINT_PREFIX = 'KFAC_'
TRANSECT_Y = [0e3, 5e3, 15e3, 18e3, 19e3, 19.5e3, 19.9e3]
TRANSECT_X = [150e3]
N_TRANSECT = 1400
INTERFACE_TRUTH_MASK_M = 2000.0


class MissingKfac:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class NoKfacUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('kfac_jax'):
            return MissingKfac
        return super().find_class(module, name)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        return NoKfacUnpickler(f).load()['params']


def region_transect(params, net, grad, eval_f, scale, idx, x_phys, y_phys):
    sc = scale[idx]
    data_mean, data_range, dynamic_scale = sc
    x_mean, y_mean, u_mean, v_mean, h_mean, s_mean = data_mean
    x_range, y_range, u_range, v_range, _, s_range = data_range
    _, _, mu_scale, _, term0, *_ = dynamic_scale

    x_n = (x_phys - x_mean) / x_range
    y_n = (y_phys - y_mean) / y_range
    X = jnp.hstack((x_n, y_n))

    U = jax.vmap(lambda z: net(params, z, idx))(X)
    G = jax.vmap(lambda z: grad(params, z, idx))(X)[:, 7:]
    H = jax.vmap(lambda z: jax.jacfwd(jax.jacfwd(lambda y: net(params, y, idx)[0:2]))(z))(X)
    residual, _ = eval_f(params, X, idx)

    u = U[:, 0:1] * u_range + u_mean
    v = U[:, 1:2] * v_range + v_mean
    h = U[:, 2:3] * h_mean
    s = U[:, 3:4] * s_range + s_mean
    mu = U[:, 4:5] * mu_scale
    ux = G[:, 0:1] * (u_range / x_range)
    uy = G[:, 1:2] * (u_range / y_range)
    vx = G[:, 2:3] * (v_range / x_range)
    vy = G[:, 3:4] * (v_range / y_range)
    mux = G[:, 8:9] * (mu_scale / x_range)
    muy = G[:, 9:10] * (mu_scale / y_range)
    uxx = H[:, 0, 0, 0:1] * (u_range / x_range**2)
    uxy = H[:, 0, 0, 1:2] * (u_range / (x_range * y_range))
    uyy = H[:, 0, 1, 1:2] * (u_range / y_range**2)
    vxx = H[:, 1, 0, 0:1] * (v_range / x_range**2)
    vxy = H[:, 1, 0, 1:2] * (v_range / (x_range * y_range))
    vyy = H[:, 1, 1, 1:2] * (v_range / y_range**2)
    stress_xx = 2.0 * h * mu * (2.0 * ux + vy)

    return {
        'x': x_phys.reshape(-1),
        'y': y_phys.reshape(-1),
        'u': u.reshape(-1),
        'v': v.reshape(-1),
        's': s.reshape(-1),
        'stress': stress_xx.reshape(-1),
        'res_x': (residual[:, 0:1] * term0).reshape(-1),
        'res_y': (residual[:, 1:2] * term0).reshape(-1),
        'h': h.reshape(-1),
        'mu': mu.reshape(-1),
        'mux': mux.reshape(-1),
        'muy': muy.reshape(-1),
        'ux': ux.reshape(-1),
        'uy': uy.reshape(-1),
        'vx': vx.reshape(-1),
        'vy': vy.reshape(-1),
        'uxx': uxx.reshape(-1),
        'uxy': uxy.reshape(-1),
        'uyy': uyy.reshape(-1),
        'vxx': vxx.reshape(-1),
        'vxy': vxy.reshape(-1),
        'vyy': vyy.reshape(-1),
    }


def ground_truth_transect(raw_region, dense_data, idx, x_phys, y_phys, axis, x_if=None):
    x_raw = raw_region[0].reshape(-1, 1)
    y_raw = raw_region[1].reshape(-1, 1)
    points = jnp.hstack((x_raw, y_raw))
    query = jnp.hstack((x_phys, y_phys))
    x_vals = np.unique(np.asarray(raw_region[0]).reshape(-1))
    y_vals = np.unique(np.asarray(raw_region[1]).reshape(-1))
    dy = np.median(np.diff(y_vals))

    u = griddata(points, raw_region[2].reshape(-1, 1), query)
    v = griddata(points, raw_region[3].reshape(-1, 1), query)
    h = griddata(points, dense_data['h_dense'][0, idx].reshape(-1, 1), query)
    s = griddata(points, dense_data['s_dense'][0, idx].reshape(-1, 1), query)
    mu = griddata(points, raw_region[12].reshape(-1, 1), query)

    x = x_phys.reshape(-1)
    y = y_phys.reshape(-1)
    u = jnp.asarray(u).reshape(-1)
    v = jnp.asarray(v).reshape(-1)
    h = jnp.asarray(h).reshape(-1)
    s = jnp.asarray(s).reshape(-1)
    mu = jnp.asarray(mu).reshape(-1)
    if axis == 'x':
        y0 = float(y_phys[0, 0])
        y_min = float(y_vals[0])
        y_max = float(y_vals[-1])
        ux = jnp.gradient(u, x)
        uy = jnp.full_like(u, jnp.nan)
        vx = jnp.gradient(v, x)
        if y0 - dy < y_min:
            query_next = jnp.hstack((x_phys, y_phys + dy))
            v_next = griddata(points, raw_region[3].reshape(-1, 1), query_next)
            vy = (jnp.asarray(v_next).reshape(-1) - v) / dy
        elif y0 + dy > y_max:
            query_prev = jnp.hstack((x_phys, y_phys - dy))
            v_prev = griddata(points, raw_region[3].reshape(-1, 1), query_prev)
            vy = (v - jnp.asarray(v_prev).reshape(-1)) / dy
        else:
            query_plus = jnp.hstack((x_phys, y_phys + dy))
            query_minus = jnp.hstack((x_phys, y_phys - dy))
            v_plus = griddata(points, raw_region[3].reshape(-1, 1), query_plus)
            v_minus = griddata(points, raw_region[3].reshape(-1, 1), query_minus)
            vy = (jnp.asarray(v_plus).reshape(-1) - jnp.asarray(v_minus).reshape(-1)) / (2.0 * dy)
        mux = jnp.gradient(mu, x)
        muy = jnp.full_like(mu, jnp.nan)
    else:
        uy = jnp.gradient(u, y)
        vy = jnp.gradient(v, y)
        muy = jnp.gradient(mu, y)
        ux = jnp.full_like(u, jnp.nan)
        vx = jnp.full_like(v, jnp.nan)
        mux = jnp.full_like(mu, jnp.nan)
    stress_xx = 2.0 * h * mu * (2.0 * ux + vy)
    if axis == 'x' and x_if is not None:
        interface_mask = jnp.abs(x - x_if) <= INTERFACE_TRUTH_MASK_M
        ux = jnp.where(interface_mask, jnp.nan, ux)
        vx = jnp.where(interface_mask, jnp.nan, vx)
        mux = jnp.where(interface_mask, jnp.nan, mux)
        stress_xx = jnp.where(interface_mask, jnp.nan, stress_xx)

    return {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        's': s,
        'stress': stress_xx,
        'h': h,
        'mu': mu,
        'mux': mux,
        'muy': muy,
        'ux': ux,
        'uy': uy,
        'vx': vx,
        'vy': vy,
    }


def values_arg(values):
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def safe_name(value):
    return str(value).replace(os.sep, '_')


def coord_name(value):
    return f'{value:g}'.replace('-', 'm').replace('.', 'p')


def interface_x_at_y(raw, y_phys):
    x_md = np.asarray(raw[0][10]).reshape(-1)
    y_md = np.asarray(raw[0][11]).reshape(-1)
    valid = np.isfinite(x_md) & np.isfinite(y_md)
    order = np.argsort(y_md[valid])
    return float(np.interp(y_phys, y_md[valid][order], x_md[valid][order]))


def x_transect_points(raw, x_if, n, y_phys):
    return [
        (jnp.linspace(float(jnp.nanmin(raw[0][0])), x_if, n).reshape(-1, 1),
         jnp.full((n, 1), y_phys)),
        (jnp.linspace(x_if, float(jnp.nanmax(raw[1][0])), n).reshape(-1, 1),
         jnp.full((n, 1), y_phys)),
    ]


def clip_x_transect_y(raw, y_phys):
    y_min = max(float(jnp.nanmin(region[1])) for region in raw)
    y_max = min(float(jnp.nanmax(region[1])) for region in raw)
    return min(max(y_phys, y_min), y_max)


def y_transect_points(raw_region, n, x_phys):
    return (
        jnp.full((n, 1), x_phys),
        jnp.linspace(float(jnp.nanmin(raw_region[1])), float(jnp.nanmax(raw_region[1])), n).reshape(-1, 1),
    )


def plot_transect(fig_path, title, xlabel, coord_key, transect, series, x_if=None):
    fig, axs = plt.subplots(len(series), 1, figsize=(8.4, 2.2 * len(series) + 1.0), sharex=True)
    axs = np.atleast_1d(axs)
    colors = ['tan', 'steelblue']
    labels = ['Region 0', 'Region 1']

    for ax, (key, ylabel, factor) in zip(axs, series):
        for i, res in enumerate(transect['results']):
            ax.plot(res[coord_key] / 1e3, res[key] * factor,
                    color=colors[i], lw=1.9, label=labels[i], zorder=2)
        for i, res in enumerate(transect['truth']):
            if key in res:
                ax.plot(res[coord_key] / 1e3, res[key] * factor,
                        color='0.15', lw=1.25, ls='--', alpha=0.7,
                        label='Ground truth' if i == 0 else None, zorder=3)
        if x_if is not None:
            ax.axvline(x_if / 1e3, color='crimson', lw=0.9, ls=':', alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)

    axs[0].legend(loc='best', fontsize=8)
    axs[-1].set_xlabel(xlabel)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f'finished plot: {fig_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=LOAD_STEP)
    parser.add_argument('--test-folder', type=Path, default=SCRIPT_DIR)
    parser.add_argument('--test-name', default=TEST_NAME)
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_FOLDER)
    parser.add_argument('--y', type=float, nargs='+', default=TRANSECT_Y)
    parser.add_argument('--x', type=float, nargs='+', default=TRANSECT_X)
    parser.add_argument('--n', type=int, default=N_TRANSECT)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    ckpt_path = args.test_folder / args.checkpoint_dir / f'{CHECKPOINT_PREFIX}step_{args.step}.pkl'
    data_path = args.test_folder / f'{args.test_name}_data_xpinns_regression_test.mat'
    plot_dir = args.test_folder / 'transect_plots' / f'{safe_name(args.checkpoint_dir)}_step_{args.step}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output.stem if args.output else f'transect_step_{args.step}'
    output_suffix = args.output.suffix if args.output else '.png'

    params = load_checkpoint(ckpt_path)
    keys = random.PRNGKey(8132002)
    data, dense_data = load_data(str(data_path), load_extra=True)
    _, xpinn = initialize_xpinn(keys, data)
    net, grad = xpinn.sol_NN
    eval_f = xpinn.eval_f
    scale = data.scale
    raw = [data.data_all[i][4][3] for i in data.idxgall]

    x_transects = []
    for y_phys in values_arg(args.y):
        y_eval = clip_x_transect_y(raw, y_phys)
        x_if = interface_x_at_y(raw, y_eval)
        label = f'y={y_phys:g} m'
        if y_eval != y_phys:
            label = f'y={y_phys:g} m (using y={y_eval:g} m)'
        points = x_transect_points(raw, x_if, args.n, y_eval)
        x_transects.append({
            'label': label,
            'coord': y_phys,
            'x_if': x_if,
            'results': [
                region_transect(params, net, grad, eval_f, scale, idx, points[idx][0], points[idx][1])
                for idx in data.idxgall
            ],
            'truth': [
                ground_truth_transect(raw[idx], dense_data, idx, points[idx][0], points[idx][1], 'x', x_if=x_if)
                for idx in data.idxgall
            ],
        })

    y_transects = []
    for x_phys in values_arg(args.x):
        region_ids = [
            idx for idx in data.idxgall
            if float(jnp.nanmin(raw[idx][0])) <= x_phys <= float(jnp.nanmax(raw[idx][0]))
        ]
        if not region_ids:
            print(f'skipping x={x_phys:g} m; outside data extent')
            continue
        y_points = [y_transect_points(raw[idx], args.n, x_phys) for idx in region_ids]
        y_transects.append({
            'label': f'x={x_phys:g} m',
            'coord': x_phys,
            'results': [
                region_transect(params, net, grad, eval_f, scale, idx, y_points[i][0], y_points[i][1])
                for i, idx in enumerate(region_ids)
            ],
            'truth': [
                ground_truth_transect(raw[idx], dense_data, idx, y_points[i][0], y_points[i][1], 'y')
                for i, idx in enumerate(region_ids)
            ],
        })

    x_series = [
        ('mu', r'$\mu$ [Pa s]', 1.0),
        ('h', r'$h$ [m]', 1.0),
        ('mux', r'$\mu_x$ [Pa s m$^{-1}$]', 1.0),
        ('ux', r'$u_x$ [s$^{-1}$]', 1.0),
        ('vx', r'$v_x$ [s$^{-1}$]', 1.0),
        ('stress', r'$hR_{xx}$ [kPa]', 1e-3),
    ]
    y_series = [
        ('mu', r'$\mu$ [Pa s]', 1.0),
        ('h', r'$h$ [m]', 1.0),
        ('muy', r'$\mu_y$ [Pa s m$^{-1}$]', 1.0),
        ('uy', r'$u_y$ [s$^{-1}$]', 1.0),
        ('vy', r'$v_y$ [s$^{-1}$]', 1.0),
    ]
    second_derivative_series = [
        ('uxx', r'$u_{xx}$ [m$^{-1}$ s$^{-1}$]', 1.0),
        ('uxy', r'$u_{xy}$ [m$^{-1}$ s$^{-1}$]', 1.0),
        ('uyy', r'$u_{yy}$ [m$^{-1}$ s$^{-1}$]', 1.0),
        ('vxx', r'$v_{xx}$ [m$^{-1}$ s$^{-1}$]', 1.0),
        ('vxy', r'$v_{xy}$ [m$^{-1}$ s$^{-1}$]', 1.0),
        ('vyy', r'$v_{yy}$ [m$^{-1}$ s$^{-1}$]', 1.0),
    ]
    state_series = [
        ('u', r'$u$ [m yr$^{-1}$]', 1.0),
        ('v', r'$v$ [m yr$^{-1}$]', 1.0),
        ('s', r'$s$ [m]', 1.0),
        ('h', r'$h$ [m]', 1.0),
        ('mu', r'$\mu$ [Pa s]', 1.0),
    ]
    residual_series = [
        ('res_x', r'$r_x\,\mathrm{term}_0$ [kPa]', 1e-3),
        ('res_y', r'$r_y\,\mathrm{term}_0$ [kPa]', 1e-3),
    ]

    for transect in x_transects:
        fig_path = plot_dir / f'{output_base}_along_x_y_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Horizontal Transect {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'x [km]',
            'x',
            transect,
            x_series,
            x_if=transect['x_if'],
        )
        fig_path = plot_dir / f'{output_base}_state_along_x_y_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Horizontal State Variables {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'x [km]',
            'x',
            transect,
            state_series,
            x_if=transect['x_if'],
        )
        fig_path = plot_dir / f'{output_base}_second_derivatives_along_x_y_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Horizontal Velocity Second Derivatives {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'x [km]',
            'x',
            transect,
            second_derivative_series,
            x_if=transect['x_if'],
        )
        fig_path = plot_dir / f'{output_base}_residual_along_x_y_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Horizontal Equation Residuals {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'x [km]',
            'x',
            transect,
            residual_series,
            x_if=transect['x_if'],
        )
        print(transect['label'])
        for key, _, factor in x_series:
            left = float(transect['results'][0][key][-1] * factor)
            right = float(transect['results'][1][key][0] * factor)
            print(f'{key}_interface_diff={left - right:.9e}')

    for transect in y_transects:
        fig_path = plot_dir / f'{output_base}_along_y_x_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Vertical Transect {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'y [km]',
            'y',
            transect,
            y_series,
        )
        fig_path = plot_dir / f'{output_base}_state_along_y_x_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Vertical State Variables {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'y [km]',
            'y',
            transect,
            state_series,
        )
        fig_path = plot_dir / f'{output_base}_second_derivatives_along_y_x_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Vertical Velocity Second Derivatives {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'y [km]',
            'y',
            transect,
            second_derivative_series,
        )
        fig_path = plot_dir / f'{output_base}_residual_along_y_x_{coord_name(transect["coord"])}m{output_suffix}'
        plot_transect(
            fig_path,
            f'Vertical Equation Residuals {transect["label"]}, {CHECKPOINT_PREFIX}step {args.step}',
            'y [km]',
            'y',
            transect,
            residual_series,
        )


if __name__ == '__main__':
    main()
