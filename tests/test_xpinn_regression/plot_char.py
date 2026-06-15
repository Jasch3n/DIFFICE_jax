import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, Delaunay


SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / 'char_outputs' / 'calc_char_region_grounded.npz'
PLOT_DPI = 220


def load_result(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def pq_clip_patch(ax, result):
    if 'pq_x' in result and 'pq_y' in result:
        x = np.asarray(result['pq_x'], dtype=float) / 1e3
        y = np.asarray(result['pq_y'], dtype=float) / 1e3
    else:
        x = np.asarray(result['vel_x'], dtype=float) / 1e3
        y = np.asarray(result['vel_y'], dtype=float) / 1e3
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    patch = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, transform=ax.transData)
    return patch


def support_points(result):
    if 'pq_x' in result and 'pq_y' in result:
        x = np.asarray(result['pq_x'], dtype=float) / 1e3
        y = np.asarray(result['pq_y'], dtype=float) / 1e3
        if x.ndim == 2 and y.ndim == 2:
            x = x.ravel()
            y = y.ravel()
    else:
        x = np.asarray(result['vel_x'], dtype=float) / 1e3
        y = np.asarray(result['vel_y'], dtype=float) / 1e3
    pts = np.column_stack([x, y])
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def domain_support_mask(result, X, Y, radius_km=0.75):
    pts = support_points(result)
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return np.ones_like(X, dtype=bool)
    query = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    tree = cKDTree(pts)
    dist, _ = tree.query(query, k=1)
    near = dist <= radius_km
    if pts.shape[0] < 3:
        return near.reshape(X.shape)
    hull = Delaunay(pts)
    inside = hull.find_simplex(query) >= 0
    return (near & inside).reshape(X.shape)


def add_pq_quiver(ax, result, stride=20, scale=1.8):
    if 'pq_x' in result and 'pq_y' in result and 'p_vel' in result and 'q_vel' in result:
        xq = np.asarray(result['pq_x'])[::stride, ::stride] / 1e3
        yq = np.asarray(result['pq_y'])[::stride, ::stride] / 1e3
        pq = np.asarray(result['p_vel'])[::stride, ::stride]
        qq = np.asarray(result['q_vel'])[::stride, ::stride]
    elif 'p_vel' in result and 'q_vel' in result:
        xq = np.asarray(result['vel_x'])[::stride] / 1e3
        yq = np.asarray(result['vel_y'])[::stride] / 1e3
        pq = np.asarray(result['p_vel'])[::stride]
        qq = np.asarray(result['q_vel'])[::stride]
    elif 'p_grid' in result and 'q_grid' in result:
        xg = np.asarray(result['x_grid'])
        yg = np.asarray(result['y_grid'])
        pg = np.asarray(result['p_grid'])
        qg = np.asarray(result['q_grid'])
        xq = xg[::stride, ::stride] / 1e3
        yq = yg[::stride, ::stride] / 1e3
        pq = pg[::stride, ::stride]
        qq = qg[::stride, ::stride]
    else:
        return
    mag = np.sqrt(pq ** 2 + qq ** 2)
    valid = np.isfinite(mag) & (mag > 0.0)
    if xq.ndim == 2:
        valid = valid & domain_support_mask(result, xq, yq)
    if not np.any(valid):
        return
    uq = np.zeros_like(pq)
    vq = np.zeros_like(qq)
    uq[valid] = pq[valid] / mag[valid]
    vq[valid] = qq[valid] / mag[valid]
    ax.quiver(
        xq[valid],
        yq[valid],
        uq[valid],
        vq[valid],
        angles='xy',
        scale_units='xy',
        pivot='tail',
        scale=scale,
        width=0.0030,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.5,
        minshaft=2.5,
        minlength=0.0,
        color='crimson',
        alpha=0.95,
        zorder=6,
    )
    ax.collections[-1].set_clip_path(pq_clip_patch(ax, result))


def add_pq_streamplot(ax, result, nx=160, ny=160, density=1.05):
    if 'pq_x' in result and 'pq_y' in result and 'p_vel' in result and 'q_vel' in result:
        x = np.asarray(result['pq_x'], dtype=float) / 1e3
        y = np.asarray(result['pq_y'], dtype=float) / 1e3
        p = np.asarray(result['p_vel'], dtype=float)
        q = np.asarray(result['q_vel'], dtype=float)
        mag = np.sqrt(p ** 2 + q ** 2)
        keep = np.isfinite(mag) & (mag > 0.0)
        if not np.any(keep):
            return
        pts = np.column_stack([x[keep], y[keep]])
        u = p[keep] / mag[keep]
        v = q[keep] / mag[keep]
        xg = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
        yg = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
        X, Y = np.meshgrid(xg, yg)
        Ug = griddata(pts, u, (X, Y), method='linear')
        Vg = griddata(pts, v, (X, Y), method='linear')
        Ug_n = griddata(pts, u, (X, Y), method='nearest')
        Vg_n = griddata(pts, v, (X, Y), method='nearest')
        Ug = np.where(np.isfinite(Ug), Ug, Ug_n)
        Vg = np.where(np.isfinite(Vg), Vg, Vg_n)
        mg = np.sqrt(Ug ** 2 + Vg ** 2)
        support = domain_support_mask(result, X, Y)
        Ug = np.where(support & np.isfinite(mg) & (mg > 0.0), Ug / mg, np.nan)
        Vg = np.where(support & np.isfinite(mg) & (mg > 0.0), Vg / mg, np.nan)
        sp = ax.streamplot(
            xg, yg, Ug, Vg,
            density=density,
            color='crimson',
            linewidth=0.9,
            arrowsize=0.8,
            minlength=0.15,
            maxlength=5.0,
            zorder=5,
        )
        clip = pq_clip_patch(ax, result)
        sp.lines.set_clip_path(clip)
        sp.arrows.set_clip_path(clip)
        return
    if 'p_vel' not in result or 'q_vel' not in result:
        return
    x = np.asarray(result['vel_x'], dtype=float) / 1e3
    y = np.asarray(result['vel_y'], dtype=float) / 1e3
    p = np.asarray(result['p_vel'], dtype=float)
    q = np.asarray(result['q_vel'], dtype=float)
    mag = np.sqrt(p ** 2 + q ** 2)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag) & (mag > 0.0)
    if not np.any(valid):
        return
    u = p[valid] / mag[valid]
    v = q[valid] / mag[valid]
    pts = np.column_stack([x[valid], y[valid]])
    xg = np.linspace(180.0, 220.0, nx)
    yg = np.linspace(-30.0, 30.0, ny)
    X, Y = np.meshgrid(xg, yg)
    Ug = griddata(pts, u, (X, Y), method='linear')
    Vg = griddata(pts, v, (X, Y), method='linear')
    Ug_n = griddata(pts, u, (X, Y), method='nearest')
    Vg_n = griddata(pts, v, (X, Y), method='nearest')
    Ug = np.where(np.isfinite(Ug), Ug, Ug_n)
    Vg = np.where(np.isfinite(Vg), Vg, Vg_n)
    m = np.sqrt(Ug ** 2 + Vg ** 2)
    keep = np.isfinite(m) & (m > 0.0) & domain_support_mask(result, X, Y)
    if not np.any(keep):
        return
    Ug = np.where(keep, Ug / m, np.nan)
    Vg = np.where(keep, Vg / m, np.nan)
    sp = ax.streamplot(
        xg, yg, Ug, Vg,
        density=density,
        color='crimson',
        linewidth=0.9,
        arrowsize=0.8,
        minlength=0.15,
        maxlength=5.0,
        zorder=5,
    )
    clip = pq_clip_patch(ax, result)
    sp.lines.set_clip_path(clip)
    sp.arrows.set_clip_path(clip)


def curve_value_panel(ax, x, y, curve_id, values, cmap, clim=None):
    if clim is None:
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
            if np.isclose(vmin, vmax):
                delta = max(abs(vmin) * 1e-6, 1.0)
                vmin -= delta
                vmax += delta
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=float(clim[0]), vmax=float(clim[1]))

    for cid in np.unique(curve_id):
        mask = curve_id == cid
        if np.sum(mask) == 0:
            continue
        xc = x[mask] / 1e3
        yc = y[mask] / 1e3
        vc = values[mask]
        if xc.size == 1:
            ax.scatter(xc, yc, c=vc, cmap=cmap, norm=norm, s=28.0, zorder=5, linewidths=0)
        else:
            points = np.column_stack([xc, yc])
            seg = np.stack([points[:-1], points[1:]], axis=1)
            seg_val = 0.5 * (vc[:-1] + vc[1:])
            lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=2.6, alpha=0.98, zorder=4)
            lc.set_array(seg_val)
            ax.add_collection(lc)
            ax.scatter(xc, yc, c=vc, cmap=cmap, norm=norm, s=10.0, zorder=5, linewidths=0)
            ax.scatter(xc[:1], yc[:1], c=vc[:1], cmap=cmap, norm=norm, s=26.0, zorder=6, linewidths=0)
            ax.scatter(xc[-1:], yc[-1:], c=vc[-1:], cmap=cmap, norm=norm, s=18.0, zorder=6, linewidths=0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def plot_characteristics(result, output_path):
    fig, axs = plt.subplots(
        1, 3, figsize=(14.8, 5.8), constrained_layout=True,
        gridspec_kw={'width_ratios': [1.45, 1.0, 1.0]},
    )
    panels = [
        ('Characteristic Curves', None, None, None),
        ('Retrieved Viscosity Along Curves', result['mu'], 'magma', (1e13, 0.75e14)),
        ('Recovered Friction Along Curves', np.log10(result['beta_curve']), 'jet', (9, 11.5))
    ]
    x = result['x']
    y = result['y']
    curve_id = result['curve_id']
    seeds_x = result['seed_x']
    seeds_y = result['seed_y']
    vel_x = result['vel_x']
    vel_y = result['vel_y']

    for ax, (title, values, cmap, clim) in zip(axs, panels):
        bg_alpha = 0.25 if values is None else 0.10
        ax.scatter(vel_x / 1e3, vel_y / 1e3, s=1.0, c='0.85', alpha=bg_alpha, linewidths=0)
        if values is None:
            if 'beta_grid' in result and 'x_grid' in result and 'y_grid' in result:
                ax.pcolormesh(result['x_grid'] / 1e3, result['y_grid'] / 1e3, result['beta_grid'],
                              shading='auto', cmap='Greys', alpha=0.25)
            add_pq_streamplot(ax, result)
            add_pq_quiver(ax, result)
            for cid in np.unique(curve_id):
                mask = curve_id == cid
                ax.plot(x[mask] / 1e3, y[mask] / 1e3, color='steelblue', lw=1.4, alpha=0.95)
        else:
            sm = curve_value_panel(ax, x, y, curve_id, values, cmap, clim=clim)
            fig.colorbar(sm, ax=ax, shrink=0.82)
        if values is None:
            ax.scatter(seeds_x / 1e3, seeds_y / 1e3, s=4.0, c='k', zorder=7)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('x [km]')
        ax.set_ylim(-30, 30)
        ax.set_xlim(180, 220)
        if values is None:
            ax.set_aspect('equal', adjustable='box')

    axs[0].set_ylabel('y [km]')
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_velocity_diagnostics(result, output_path):
    x = result['vel_x'] / 1e3
    y = result['vel_y'] / 1e3
    panels = [
        ('u ground truth', result['u_obs'], 'viridis'),
        ('u surrogate', result['u_pred'], 'viridis'),
        ('u misfit', result['u_misfit'], 'coolwarm'),
        ('v ground truth', result['v_obs'], 'viridis'),
        ('v surrogate', result['v_pred'], 'viridis'),
        ('v misfit', result['v_misfit'], 'coolwarm'),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, sharey=True)
    for ax, (title, values, cmap) in zip(axs.reshape(-1), panels):
        sc = ax.scatter(x, y, c=values, s=3.0, cmap=cmap, alpha=0.9, linewidths=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, shrink=0.82)
    for ax in axs[-1, :]:
        ax.set_xlabel('x [km]')
    for ax in axs[:, 0]:
        ax.set_ylabel('y [km]')
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--velocity-output', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = load_result(args.input)
    output = args.output if args.output is not None else args.input.with_suffix('.png')
    velocity_output = (
        args.velocity_output
        if args.velocity_output is not None
        else args.input.with_name(args.input.stem + '_velocity.png')
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    velocity_output.parent.mkdir(parents=True, exist_ok=True)
    plot_characteristics(result, output)
    plot_velocity_diagnostics(result, velocity_output)
    print(f'finished plot: {output}')
    print(f'finished velocity plot: {velocity_output}')


if __name__ == '__main__':
    main()
