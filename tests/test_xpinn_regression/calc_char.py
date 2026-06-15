import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import griddata

from diffice_jax import solu_xpinn

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'char_outputs'
DEFAULT_DATA = SCRIPT_DIR / 'subglacial_channel_data_xpinns_regression_test.mat'
DEFAULT_CKPT = SCRIPT_DIR / 'match_ct_eqn_subglacial_channel_checkpoints' / 'KFAC_step_9000.pkl'
SEED_COUNT, S_MAX, DS, EPS, RHO_I, G = 32, 5.0e4, 50.0, 1e-12, 917.0, 9.8
LEARNING_RATE = 1.0e-3


class MissingKfac:
    def __setstate__(self, state): self.__dict__.update(state if isinstance(state, dict) else {})


class NoKfacUnpickler(pickle.Unpickler):
    def find_class(self, module, name): return MissingKfac if module.startswith('kfac_jax') else super().find_class(module, name)


def load_params(path):
    with open(path, 'rb') as f: ckpt = NoKfacUnpickler(f).load()
    print(f'[calc_char] loaded checkpoint: {path} (step {ckpt.get("step", "unknown")})')
    return jax.device_put(ckpt['params'])


def denorm_xy(xy_n, scale):
    dm, dr = scale.data_mean, scale.data_range
    return np.column_stack([xy_n[:, 0] * dr.x_range + dm.x_mean, xy_n[:, 1] * dr.y_range + dm.y_mean])


def grounded_region(data_output, region=None):
    grounded = [i for i, basal in zip(data_output.idxgall, data_output.basal_mask) if basal]
    idx = grounded[0] if region is None else int(region)
    data, scale = data_output.data_all[idx], data_output.scale[idx]
    raw, (idx_u, idx_h), dsize = data[4][3], data[4][4], tuple(int(v) for v in data[4][5][0])
    vel_xy = np.column_stack([np.asarray(raw[0]).reshape(-1)[idx_u], np.asarray(raw[1]).reshape(-1)[idx_u]])
    thk_xy = np.column_stack([np.asarray(raw[4]).reshape(-1)[idx_h], np.asarray(raw[5]).reshape(-1)[idx_h]])
    x_all, y_all = np.r_[vel_xy[:, 0], thk_xy[:, 0]], np.r_[vel_xy[:, 1], thk_xy[:, 1]]
    seed_xy_n = np.asarray(data[2])
    if seed_xy_n.ndim == 2 and seed_xy_n.shape[1] >= 2 and seed_xy_n.shape[0] > 0:
        seed_xy = denorm_xy(seed_xy_n, scale)
    else:
        seed_xy = vel_xy[:max(1, min(vel_xy.shape[0], 32))]
    if len(data) > 6 and data[6] is not None and len(data[6]) > 0 and np.asarray(data[6][0]).size > 0:
        seed_mu = np.asarray(data[6][0]).reshape(-1, 1) * float(scale.dynamic_scale.mu0)
    elif len(data[1]) > 3:
        seed_mu = np.asarray(data[1][3]).reshape(-1, 1)[:seed_xy.shape[0]] * float(scale.dynamic_scale.mu0)
    else:
        seed_mu = np.ones((seed_xy.shape[0], 1), dtype=float) * float(scale.dynamic_scale.mu0)
    return dict(
        idx=idx, scale=scale, vel_xy=vel_xy, thk_xy=thk_xy,
        vel_u=np.asarray(raw[2]).reshape(-1)[idx_u], vel_v=np.asarray(raw[3]).reshape(-1)[idx_u],
        thk_h=np.asarray(raw[6]).reshape(-1)[idx_h], thk_s=np.asarray(raw[7]).reshape(-1)[idx_h],
        seed_xy=seed_xy,
        seed_mu=seed_mu,
        bounds=np.array([[np.nanmin(x_all), np.nanmax(x_all)], [np.nanmin(y_all), np.nanmax(y_all)]], dtype=float),
        x_grid=np.asarray(raw[0]).reshape(dsize), y_grid=np.asarray(raw[1]).reshape(dsize),
    )


def pq_terms_from_kinematics(u, v, ux, uy, vx, vy):
    exx2 = 4.0 * ux + 2.0 * vy
    exy = uy + vx
    eyy2 = 4.0 * vy + 2.0 * ux
    return v * exx2 - u * exy, v * exy - u * eyy2, exx2, exy, eyy2


def rg_terms_from_kinematics(u, v, h, sx, sy, ux, uy, vx, vy, div1, div2):
    p, q, _, _, _ = pq_terms_from_kinematics(u, v, ux, uy, vx, vy)
    r = v * div1 - u * div2
    g = RHO_I * G * h * (v * sx - u * sy)
    return p, q, r, g


def trace_characteristic(seed_xy, seed_mu, local_eval, bounds, s_max, ds, unit_speed=True):
    steps = int(np.floor(s_max / ds))
    state = np.array([seed_xy[0], seed_xy[1], float(local_eval(seed_xy)['h']) * float(seed_mu)], dtype=float)
    curve = []
    for step in range(steps + 1):
        vals = local_eval(state[:2])
        h = float(vals['h'])
        curve.append(dict(
            s=step * ds,
            x=state[0],
            y=state[1],
            hnu=state[2],
            h=h,
            mu=state[2] / (h + EPS),
            u=float(vals['u']),
            v=float(vals['v']),
        ))
        if step == steps:
            break
        p, q, r, g = float(vals['p']), float(vals['q']), float(vals['r']), float(vals['g'])
        if unit_speed:
            scale = max((p * p + q * q) ** 0.5, EPS)
            p, q, r, g = p / scale, q / scale, r / scale, g / scale
        state = state + ds * np.array([p, q, g - r * state[2]], dtype=float)
        if not ((bounds[0, 0] <= state[0] <= bounds[0, 1]) and (bounds[1, 0] <= state[1] <= bounds[1, 1])):
            break
    return curve


def extract_grounded_region(data_output):
    region = grounded_region(data_output)
    region['region_idx'] = region['idx']
    return region


def run_characteristics(data_output, key, steps=5, seed_count=3, s_max=4.0e3, ds=2.0e3, width=8, depth=1):
    region = extract_grounded_region(data_output)
    seeds = region['seed_xy'][:seed_count]
    seed_mu = region['seed_mu'][:seed_count, 0]

    def local_eval(_xy):
        return dict(p=1.0, q=0.0, r=0.0, g=1.0, u=1.0, v=0.0, h=1.0)

    curves = [
        trace_characteristic(seed, mu, local_eval, region['bounds'], s_max=s_max, ds=ds, unit_speed=True)
        for seed, mu in zip(seeds, seed_mu)
    ]
    flat_curves = flatten(curves)
    hnu_grid = np.full(region['x_grid'].shape, np.nanmean(flat_curves['hnu']))
    beta_grid = np.zeros(region['x_grid'].shape)
    return dict(
        grounded=region,
        curves=curves,
        flat_curves=flat_curves,
        hnu_grid=hnu_grid,
        beta_grid=beta_grid,
    )


def save_results(output, result, args):
    flat = result['flat_curves']
    np.savez(
        output,
        x=flat['x'],
        y=flat['y'],
        hnu=flat['hnu'],
        hnu_grid=result['hnu_grid'],
        beta_grid=result['beta_grid'],
        steps=args.steps,
        lr=args.lr,
        seed_count=args.seed_count,
        s_max=args.s_max,
        ds=args.ds,
    )


def normalize_xy(xy, scale):
    dm, dr = scale.data_mean, scale.data_range
    return jnp.array([(xy[0] - dm.x_mean) / dr.x_range, (xy[1] - dm.y_mean) / dr.y_range])


def make_local_eval(params, scales, basal_mask, idx):
    pred_u, _ = solu_xpinn(scales, basal_mask=basal_mask)
    scale, dm, dr, dyn = scales[idx], scales[idx].data_mean, scales[idx].data_range, scales[idx].dynamic_scale

    def sol(xy):
        out = pred_u(params, normalize_xy(xy, scale), idx)
        return jnp.array([out[0] * dr.u_range + dm.u_mean, out[1] * dr.v_range + dm.v_mean,
                          out[2] * dm.h_mean, out[3] * dr.s_range + dm.s_mean,
                          out[4] * dyn.mu0, out[5] * dyn.c0])

    def f(xy):
        y, g, hu, hv = sol(xy), jax.jacfwd(sol)(xy), jax.hessian(lambda z: sol(z)[0])(xy), jax.hessian(lambda z: sol(z)[1])(xy)
        u, v, h, s, mu, c = y
        ux, uy, vx, vy, hx, hy, sx, sy, mux, muy = g[0, 0], g[0, 1], g[1, 0], g[1, 1], g[2, 0], g[2, 1], g[3, 0], g[3, 1], g[4, 0], g[4, 1]
        a, b, d = 4 * ux + 2 * vy, uy + vx, 4 * vy + 2 * ux
        div1, div2 = 4 * hu[0, 0] + 2 * hv[1, 0] + hu[1, 1] + hv[0, 1], hu[0, 1] + hv[0, 0] + 4 * hv[1, 1] + 2 * hu[0, 1]
        p, q, r, gg = v * a - u * b, v * b - u * d, v * div1 - u * div2, RHO_I * G * h * (v * sx - u * sy)
        return dict(u=u, v=v, h=h, s=s, mu=mu, c=c, ux=ux, uy=uy, vx=vx, vy=vy, hx=hx, hy=hy, sx=sx, sy=sy,
                    mux=mux, muy=muy, a=a, b=b, d=d, div1=div1, div2=div2, p=p, q=q, r=r, g=gg)

    return jax.jit(f)


def choose_dir(seed_xy, local_eval, ds, centroid):
    vals = local_eval(jnp.asarray(seed_xy)); p, q = float(vals['p']), float(vals['q']); n = max((p * p + q * q) ** 0.5, EPS)
    plus, minus = seed_xy + ds * np.array([p / n, q / n]), seed_xy - ds * np.array([p / n, q / n])
    return 1.0 if np.linalg.norm(plus - centroid) <= np.linalg.norm(minus - centroid) else -1.0


def make_integrator(local_eval, bounds, ds, s_max):
    bounds, steps = jnp.asarray(bounds), int(np.floor(s_max / ds))

    @jax.jit
    def run(state0, direction):
        def rhs(state):
            vals = local_eval(state[:2]); p, q, r, g = direction * vals['p'], direction * vals['q'], direction * vals['r'], direction * vals['g']
            n = jnp.maximum(jnp.sqrt(p * p + q * q), EPS)
            return jnp.array([p / n, q / n, (g - r * state[2]) / n])

        def rk4(state):
            k1 = rhs(state); k2 = rhs(state + 0.5 * ds * k1); k3 = rhs(state + 0.5 * ds * k2); k4 = rhs(state + ds * k3)
            return state + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def body(carry, i):
            state, active = carry; vals = local_eval(state[:2]); nxt = rk4(state)
            inside = (nxt[0] >= bounds[0, 0]) & (nxt[0] <= bounds[0, 1]) & (nxt[1] >= bounds[1, 0]) & (nxt[1] <= bounds[1, 1])
            keep = active & jnp.all(jnp.isfinite(jnp.array([*state, vals['h'], vals['u'], vals['v']]))) & jnp.all(jnp.isfinite(nxt)) & inside
            row = jnp.where(keep, jnp.array([i * ds, state[0], state[1], state[2], vals['h'], state[2] / (vals['h'] + EPS), vals['u'], vals['v']]), jnp.full((8,), jnp.nan))
            return (jnp.where(keep, nxt, state), keep), row

        return jax.lax.scan(body, (state0, jnp.array(True)), jnp.arange(1, steps + 1))

    return run


def trace(seed_xy, seed_mu, local_eval, integrator, bounds, centroid, curve_idx):
    first, direction = local_eval(jnp.asarray(seed_xy)), choose_dir(seed_xy, local_eval, DS, centroid)
    print(f'[calc_char] curve {curve_idx:>4d} start | seed=({seed_xy[0]:.1f}, {seed_xy[1]:.1f}) | mu0={float(seed_mu):.3e} | dir={direction:+.0f}')
    inward = centroid - seed_xy; inward = inward / max(np.linalg.norm(inward), EPS); state0 = jnp.array([seed_xy[0] + 0.1 * DS * inward[0], seed_xy[1] + 0.1 * DS * inward[1], float(first["h"]) * float(seed_mu)])
    (_, active), rows = integrator(state0, jnp.asarray(direction)); rows = np.asarray(rows); rec = rows[np.isfinite(rows[:, 0])]
    out = [dict(s=0.0, x=seed_xy[0], y=seed_xy[1], hnu=float(first['h']) * float(seed_mu), h=float(first['h']), mu=float(seed_mu), u=float(first['u']), v=float(first['v']))]
    out += [dict(s=r[0], x=r[1], y=r[2], hnu=r[3], h=r[4], mu=r[5], u=r[6], v=r[7]) for r in rec]
    last = out[-1]; reason = 's_max' if bool(np.asarray(active)) else 'bbox'
    print(f'[calc_char] curve {curve_idx:>4d} done  | points={len(out):>4d} | s_end={last["s"]:.1f} | end=({last["x"]:.1f}, {last["y"]:.1f}) | reason={reason}')
    return out


def flatten(curves):
    rows = [[i, r['s'], r['x'], r['y'], r['hnu'], r['h'], r['mu'], r['u'], r['v']] for i, c in enumerate(curves) for r in c]
    arr = np.asarray(rows, dtype=float)
    return dict(curve_id=arr[:, 0].astype(int), s=arr[:, 1], x=arr[:, 2], y=arr[:, 3], hnu=arr[:, 4], h=arr[:, 5], nu=arr[:, 6], u=arr[:, 7], v=arr[:, 8])


def recover_beta(local_eval, x_grid, y_grid, flat):
    hnu_grid = griddata(np.column_stack([flat['x'], flat['y']]), flat['hnu'], (x_grid, y_grid), method='nearest')
    pts = jnp.asarray(np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1)])); vals = jax.vmap(local_eval)(pts)
    fields = {k: np.asarray(v).reshape(x_grid.shape) for k, v in vals.items()}
    x_axis, y_axis = np.asarray(x_grid[0, :]), np.asarray(y_grid[:, 0])
    hnu_x = np.zeros_like(hnu_grid) if hnu_grid.shape[1] <= 1 or x_axis.size <= 1 else np.gradient(hnu_grid, x_axis, axis=1)
    hnu_y = np.zeros_like(hnu_grid) if hnu_grid.shape[0] <= 1 or y_axis.size <= 1 else np.gradient(hnu_grid, y_axis, axis=0)
    l1, l2 = fields['a'] * hnu_x + fields['b'] * hnu_y + fields['div1'] * hnu_grid, fields['b'] * hnu_x + fields['d'] * hnu_y + fields['div2'] * hnu_grid
    beta_grid = (fields['u'] * (l1 - RHO_I * G * fields['h'] * fields['sx']) + fields['v'] * (l2 - RHO_I * G * fields['h'] * fields['sy'])) / (fields['u'] ** 2 + fields['v'] ** 2 + EPS)
    beta_curve = griddata(np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1)]), beta_grid.reshape(-1), np.column_stack([flat['x'], flat['y']]), method='nearest')
    return hnu_grid, beta_grid, beta_curve, fields


def boundary_axis(lo, hi, n=161, power=3.0):
    t = np.linspace(-1.0, 1.0, n)
    return lo + (hi - lo) * 0.5 * (1.0 + np.sign(t) * np.abs(t) ** power)


def pq_dense(local_eval, bounds, nx=161, ny=161):
    x = boundary_axis(bounds[0, 0], bounds[0, 1], nx)
    y = boundary_axis(bounds[1, 0], bounds[1, 1], ny)
    X, Y = np.meshgrid(x, y)
    vals = jax.vmap(local_eval)(jnp.asarray(np.column_stack([X.reshape(-1), Y.reshape(-1)])))
    return X, Y, np.asarray(vals['p']).reshape(X.shape), np.asarray(vals['q']).reshape(X.shape)


def save(path, region, fit_rmse, flat, curves, hnu_grid, beta_grid, beta_curve, fields, pq_x, pq_y, p_vel, q_vel):
    np.savez(path, region_idx=region['idx'], curve_id=flat['curve_id'], s=flat['s'], x=flat['x'], y=flat['y'], hnu=flat['hnu'],
             h=flat['h'], nu=flat['nu'], mu=flat['nu'], u=flat['u'], v=flat['v'], beta_curve=beta_curve,
             u_obs=region['vel_u'], v_obs=region['vel_v'], u_pred=fields['u'].reshape(-1)[:region['vel_xy'].shape[0]], v_pred=fields['v'].reshape(-1)[:region['vel_xy'].shape[0]],
             u_misfit=fields['u'].reshape(-1)[:region['vel_xy'].shape[0]] - region['vel_u'], v_misfit=fields['v'].reshape(-1)[:region['vel_xy'].shape[0]] - region['vel_v'],
             vel_x=region['vel_xy'][:, 0], vel_y=region['vel_xy'][:, 1], pq_x=pq_x, pq_y=pq_y, p_vel=p_vel, q_vel=q_vel, seed_x=region['seed_xy'][:, 0], seed_y=region['seed_xy'][:, 1],
             seed_mu=region['seed_mu'][:, 0], x_grid=region['x_grid'], y_grid=region['y_grid'], p_grid=fields['p'], q_grid=fields['q'],
             hnu_grid=hnu_grid, beta_grid=beta_grid, fit_rmse_u=fit_rmse[0], fit_rmse_v=fit_rmse[1], fit_rmse_h=fit_rmse[2], fit_rmse_s=fit_rmse[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=Path, default=DEFAULT_DATA); p.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--output', type=Path, default=None); p.add_argument('--seed-count', type=int, default=SEED_COUNT)
    p.add_argument('--s-max', type=float, default=S_MAX); p.add_argument('--ds', type=float, default=DS); p.add_argument('--region', type=int, default=None)
    args = p.parse_args(); sys.path.insert(0, str(SCRIPT_DIR)); from xpinn_regression import load_data
    data_output, params = load_data(str(args.data_path)), load_params(args.checkpoint)
    region = grounded_region(data_output, args.region); local_eval = make_local_eval(params, data_output.scale, data_output.basal_mask, region['idx'])
    vel_vals, thk_pts = jax.vmap(local_eval)(jnp.asarray(region['vel_xy'])), jax.vmap(local_eval)(jnp.asarray(np.column_stack([region['x_grid'].reshape(-1), region['y_grid'].reshape(-1)])))
    fit_rmse = [float(jnp.sqrt(jnp.mean((vel_vals['u'] - region['vel_u']) ** 2))), float(jnp.sqrt(jnp.mean((vel_vals['v'] - region['vel_v']) ** 2))),
                float(jnp.nan), float(jnp.nan)]
    seed_ok = np.isfinite(region['seed_xy']).all(axis=1) & np.isfinite(region['seed_mu'][:, 0]); seeds, seed_mu = region['seed_xy'][seed_ok], region['seed_mu'][seed_ok]
    if args.seed_count < len(seeds): idx = np.linspace(0, len(seeds) - 1, args.seed_count, dtype=int); seeds, seed_mu = seeds[idx], seed_mu[idx]
    centroid = np.array([np.nanmean(region['vel_xy'][:, 0]), np.nanmean(region['vel_xy'][:, 1])], dtype=float)
    integrator = make_integrator(local_eval, region['bounds'], args.ds, args.s_max); curves = [trace(seeds[i], seed_mu[i, 0], local_eval, integrator, region['bounds'], centroid, i) for i in range(len(seeds))]
    flat = flatten(curves); hnu_grid, beta_grid, beta_curve, fields = recover_beta(local_eval, region['x_grid'], region['y_grid'], flat)
    pq_x, pq_y, p_dense, q_dense = pq_dense(local_eval, region['bounds'])
    p_vel, q_vel = np.asarray(vel_vals['p']).reshape(-1), np.asarray(vel_vals['q']).reshape(-1)
    output = args.output or (OUTPUT_DIR / f'calc_char_region_{region["idx"] if args.region is not None else "grounded"}.npz'); output.parent.mkdir(parents=True, exist_ok=True)
    save(output, region, fit_rmse, flat, curves, hnu_grid, beta_grid, beta_curve, fields, pq_x, pq_y, p_dense, q_dense)
    print(f'finished npz: {output}')


if __name__ == '__main__':
    main()
