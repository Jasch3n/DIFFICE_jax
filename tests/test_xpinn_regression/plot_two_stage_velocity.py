import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from xpinn_regression import DATA_PATH, load_data, initialize_xpinn
from jax import random


def col(x):
    return jnp.asarray(x).reshape(-1, 1)


def flat(x):
    return np.asarray(x).reshape(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-stage-checkpoint', required=True)
    parser.add_argument('--second-stage-checkpoint', required=True)
    parser.add_argument('--data-path', default=DATA_PATH)
    parser.add_argument('--output', default='two_stage_velocity.png')
    parser.add_argument('--region', type=int, default=None)
    parser.add_argument('--err-scale', type=float, default=0.02)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def load_params(path):
    ckpt_path = os.path.abspath(path)
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    if 'params' not in ckpt:
        raise ValueError(f'Checkpoint missing params: {ckpt_path}')
    print(f'Loaded checkpoint: {ckpt_path} (step {ckpt.get("step", "unknown")})')
    return jax.device_put(ckpt['params'])


def dimensionalize_velocity(u_norm, v_norm, scale):
    _, _, u_mean, v_mean, _, _ = scale.data_mean
    _, _, u_range, v_range, _, _ = scale.data_range
    u = u_norm * u_range + u_mean
    v = v_norm * v_range + v_mean
    return u, v

# [TODO]: Correctly initialize NN with fourier features 
def velocity_prediction(net, params_1st, params_2nd, x_norm, u_norm, region_idx, scale):
    pred_1st = net(params_1st, x_norm, region_idx)
    pred_2nd = net(params_2nd, x_norm, region_idx)
    residual_rms = jnp.maximum(jnp.sqrt(jnp.mean((u_norm - pred_1st[:, 0:2]) ** 2, axis=0)), 1e-12)
    pred_sum = pred_1st[:, 0:2] + residual_rms * pred_2nd[:, 0:2]

    u_1st, v_1st = dimensionalize_velocity(pred_1st[:, 0], pred_1st[:, 1], scale)
    u_sum, v_sum = dimensionalize_velocity(pred_sum[:, 0], pred_sum[:, 1], scale)
    return u_1st, v_1st, u_sum, v_sum, residual_rms


def plot_region(axs, region_idx, data_raw, data_norm, scale, net, params_1st, params_2nd, err_scale):
    x_raw = flat(data_raw[0] / 1e3)
    y_raw = flat(data_raw[1] / 1e3)
    u_true = flat(data_raw[2])
    v_true = flat(data_raw[3])
    x_norm = jnp.hstack((col(data_norm[0]), col(data_norm[1])))
    u_norm = jnp.hstack((col(data_norm[2]), col(data_norm[3])))

    u_1st, v_1st, u_pred, v_pred, residual_rms = velocity_prediction(
        net, params_1st, params_2nd, x_norm, u_norm, region_idx, scale)
    u_1st = flat(u_1st)
    v_1st = flat(v_1st)
    u_pred = flat(u_pred)
    v_pred = flat(v_pred)
    rel_err_u_1st = (u_1st - u_true) / np.maximum(np.abs(u_true), 1.0)
    rel_err_v_1st = (v_1st - v_true) / np.maximum(np.abs(v_true), 1.0)
    rel_err_u = (u_pred - u_true) / np.maximum(np.abs(u_true), 1.0)
    rel_err_v = (v_pred - v_true) / np.maximum(np.abs(v_true), 1.0)

    plots = [
        axs[0, 0].tripcolor(x_raw, y_raw, u_true, cmap='magma'),
        axs[0, 1].tripcolor(x_raw, y_raw, v_true, cmap='jet'),
        axs[1, 0].tripcolor(x_raw, y_raw, u_1st, cmap='magma'),
        axs[1, 1].tripcolor(x_raw, y_raw, v_1st, cmap='jet'),
        axs[2, 0].tripcolor(x_raw, y_raw, rel_err_u_1st, vmin=-err_scale, vmax=err_scale, cmap='RdBu_r'),
        axs[2, 1].tripcolor(x_raw, y_raw, rel_err_v_1st, vmin=-err_scale, vmax=err_scale, cmap='RdBu_r'),
        axs[3, 0].tripcolor(x_raw, y_raw, u_pred, cmap='magma'),
        axs[3, 1].tripcolor(x_raw, y_raw, v_pred, cmap='jet'),
        axs[4, 0].tripcolor(x_raw, y_raw, rel_err_u, vmin=-err_scale, vmax=err_scale, cmap='RdBu_r'),
        axs[4, 1].tripcolor(x_raw, y_raw, rel_err_v, vmin=-err_scale, vmax=err_scale, cmap='RdBu_r'),
    ]
    titles = [
        ['Ground Truth $u$ [m/yr]', 'Ground Truth $v$ [m/yr]'],
        ['Stage 1 $u$ [m/yr]', 'Stage 1 $v$ [m/yr]'],
        ['Stage 1 Rel. Err. $u$', 'Stage 1 Rel. Err. $v$'],
        ['Stage 1 + Stage 2 $u$ [m/yr]', 'Stage 1 + Stage 2 $v$ [m/yr]'],
        ['Stage 1 + Stage 2 Rel. Err. $u$', 'Stage 1 + Stage 2 Rel. Err. $v$'],
    ]
    for row in range(5):
        for ax_col in range(2):
            axs[row, ax_col].set_title(titles[row][ax_col])
            axs[row, ax_col].set_xlabel('$x$ [km]')
            axs[row, ax_col].set_ylabel('$y$ [km]')
            plt.colorbar(plots[2 * row + ax_col], ax=axs[row, ax_col])

    print(f'Region {region_idx}: stage 1 mean abs rel err u = {np.nanmean(np.abs(rel_err_u_1st))*100:.2f}%')
    print(f'Region {region_idx}: stage 1 mean abs rel err v = {np.nanmean(np.abs(rel_err_v_1st))*100:.2f}%')
    print(f'Region {region_idx}: stage 1 residual rms u = {float(residual_rms[0]):.3e}')
    print(f'Region {region_idx}: stage 1 residual rms v = {float(residual_rms[1]):.3e}')
    print(f'Region {region_idx}: stage 1 + stage 2 mean abs rel err u = {np.nanmean(np.abs(rel_err_u))*100:.2f}%')
    print(f'Region {region_idx}: stage 1 + stage 2 mean abs rel err v = {np.nanmean(np.abs(rel_err_v))*100:.2f}%')


def main():
    args = parse_args()
    params_1st = load_params(args.first_stage_checkpoint)
    params_2nd = load_params(args.second_stage_checkpoint)

    keys = random.PRNGKey(8132002)
    data = load_data(args.data_path)
    _, xpinn = initialize_xpinn(keys, data)
    net = xpinn.sol_NN[0]

    regions = [args.region] if args.region is not None else list(range(len(data.idxgall)))
    fig, axs = plt.subplots(5, 2 * len(regions), figsize=(10 * len(regions), 15), squeeze=False)

    for col, region_idx in enumerate(regions):
        data_raw = data.data_all[region_idx][4][3]
        data_norm = data.data_all[region_idx][4][2]
        region_axes = axs[:, 2 * col:2 * col + 2]
        plot_region(region_axes, region_idx, data_raw, data_norm, data.scale[region_idx],
                    net, params_1st, params_2nd, args.err_scale)
        region_axes[0, 0].text(0.02, 1.10, f'Region {region_idx}', transform=region_axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold')

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
        print(f'Saved figure to {os.path.abspath(args.output)}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
