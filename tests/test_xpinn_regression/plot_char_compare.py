import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / 'char_outputs' / 'calc_char_region_grounded.npz'
DEFAULT_DATA = SCRIPT_DIR / 'subglacial_channel_data_xpinns_regression_test.mat'
PLOT_DPI = 220


def load_result(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def sample_truth_fields(data_path, region_idx, x, y):
    raw = loadmat(data_path)
    x_src = np.asarray(raw['xd'][0, region_idx]).reshape(-1)
    y_src = np.asarray(raw['yd'][0, region_idx]).reshape(-1)
    mu_src = np.asarray(raw['mud'][0, region_idx]).reshape(-1)
    beta_src = np.asarray(raw['alpha2d'][0, region_idx]).reshape(-1)

    valid_mu = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(mu_src)
    valid_beta = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(beta_src)
    query = np.column_stack([np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)])

    mu_lin = griddata(np.column_stack([x_src[valid_mu], y_src[valid_mu]]), mu_src[valid_mu], query, method='linear')
    mu_near = griddata(np.column_stack([x_src[valid_mu], y_src[valid_mu]]), mu_src[valid_mu], query, method='nearest')
    mu_true = np.where(np.isfinite(mu_lin), mu_lin, mu_near)

    beta_lin = griddata(np.column_stack([x_src[valid_beta], y_src[valid_beta]]), beta_src[valid_beta], query, method='linear')
    beta_near = griddata(np.column_stack([x_src[valid_beta], y_src[valid_beta]]), beta_src[valid_beta], query, method='nearest')
    beta_true = np.where(np.isfinite(beta_lin), beta_lin, beta_near)
    return mu_true, beta_true


def misfit_stats(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = np.asarray(pred)[mask]
    true = np.asarray(true)[mask]
    err = pred - true
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(true), 1e-12)
    pos = (pred > 0.0) & (true > 0.0)
    log_rmse = np.sqrt(np.mean((np.log10(pred[pos]) - np.log10(true[pos])) ** 2)) if np.any(pos) else np.nan
    corr = np.corrcoef(pred, true)[0, 1] if pred.size > 1 else np.nan
    return {
        'n': int(pred.size),
        'bias': float(np.mean(err)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(abs_err)),
        'medae': float(np.median(abs_err)),
        'mare': float(np.mean(rel)),
        'log10_rmse': float(log_rmse),
        'corr': float(corr),
        'pred_min': float(np.min(pred)),
        'pred_max': float(np.max(pred)),
        'true_min': float(np.min(true)),
        'true_max': float(np.max(true)),
        'mask': mask,
    }


def print_stats(label, stats):
    print(
        f'{label}: n={stats["n"]} | '
        f'bias={stats["bias"]:.3e} | rmse={stats["rmse"]:.3e} | mae={stats["mae"]:.3e} | '
        f'medae={stats["medae"]:.3e} | mare={stats["mare"]:.3e} | '
        f'log10_rmse={stats["log10_rmse"]:.3e} | corr={stats["corr"]:.3f}'
    )
    print(
        f'       pred=[{stats["pred_min"]:.3e}, {stats["pred_max"]:.3e}] | '
        f'true=[{stats["true_min"]:.3e}, {stats["true_max"]:.3e}]'
    )


def scatter_panel(ax, truth, pred, title, color):
    mask = np.isfinite(truth) & np.isfinite(pred) & (truth > 0.0) & (pred > 0.0)
    xt = truth[mask]
    yp = pred[mask]
    ax.scatter(xt, yp, s=3.0, c=color, alpha=0.25, linewidths=0)
    if xt.size > 0:
        lo = min(float(np.min(xt)), float(np.min(yp)))
        hi = max(float(np.max(xt)), float(np.max(yp)))
        ax.plot([lo, hi], [lo, hi], color='k', lw=1.2, alpha=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel('truth')
    ax.set_ylabel('retrieved')
    ax.grid(True, alpha=0.25)


def plot_comparison(mu_true, mu_pred, beta_true, beta_pred, output_path):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0))
    scatter_panel(axs[0], mu_true, mu_pred, 'Viscosity', 'tab:orange')
    scatter_panel(axs[1], beta_true, beta_pred, 'Friction', 'tab:blue')
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--data-path', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = load_result(args.input)
    region_idx = int(np.asarray(result['region_idx']).reshape(-1)[0])
    mu_true, beta_true = sample_truth_fields(args.data_path, region_idx, result['x'], result['y'])
    mu_pred = np.asarray(result['mu']).reshape(-1)
    beta_pred = np.asarray(result['beta_curve']).reshape(-1)

    mu_stats = misfit_stats(mu_pred, mu_true)
    beta_stats = misfit_stats(beta_pred, beta_true)
    print(f'region_idx={region_idx}')
    print_stats('mu', mu_stats)
    print_stats('beta', beta_stats)

    output = args.output if args.output is not None else args.input.with_name(args.input.stem + '_compare.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(mu_true, mu_pred, beta_true, beta_pred, output)
    print(f'finished comparison plot: {output}')


if __name__ == '__main__':
    main()
