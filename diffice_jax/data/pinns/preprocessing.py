'''
@author: Yongji Wang
Goal: "preprocessing.py" normalize the synthetic data from COMSOL and
organize the data into a form that is required for the PINN training
'''

import numpy as np
import jax.numpy as jnp
from jax import lax


def normalize_data(data,basal=False):
    '''
    :param data: original dataset
    :return X_smp, U_smp, X_ct, n_ct, data_info
    '''
    rho = 917
    rho_w = 1030
    g = 9.8
    gd = g * (1 - rho / rho_w)  # gravitational acceleration

    # extract the velocity data
    xraw = data['xd']   # unit [m] position
    yraw = data['yd']   # unit [m] position
    uraw = data['ud']   # unit [m/s] ice velocity
    vraw = data['vd']   # unit [m/s] ice velocity

    # extract the thickness data (may have different position)
    xraw_h = data['xd_h']   # unit [m] position
    yraw_h = data['yd_h']   # unit [m] position
    hraw = data['hd']       # unit [m] ice thickness
    if basal:
        sraw = data['sd']

    # extract the position of the calving front (right side of the domain)
    xct = data['xct']    # unit [m] position
    yct = data['yct']    # unit [m] position
    if basal:
        print("Basal. NO nnct")
        nnct=None
    else:
        nnct = data['nnct']  # unit vector

    # extract variables at the grounding line for basal inversions
    if basal:
        uraw_gl = data['gl_ud'].flatten()
        vraw_gl = data['gl_vd'].flatten()
        muraw_gl = data['gl_mu'].flatten()
        hraw_gl = data['gl_hd'].flatten()
        xdraw_walls = data['xd_walls'].flatten()
        ydraw_walls = data['yd_walls'].flatten()

    #%%

    # flatten the velocity data into 1d array
    x0 = xraw.flatten()
    y0 = yraw.flatten()
    u0 = uraw.flatten()
    v0 = vraw.flatten()

    # flatten the thickness data into 1d array
    x0_h = xraw_h.flatten()
    y0_h = yraw_h.flatten()
    h0 = hraw.flatten()
    if basal: 
        s0 = sraw.flatten()

    # remove the nan value in the velocity data
    idxval_u = jnp.where(~np.isnan(u0))[0]
    x = x0[idxval_u, None]
    y = y0[idxval_u, None]
    u = u0[idxval_u, None]
    v = v0[idxval_u, None]
    # if basal:
    #     ocean_mask[idxval_u, None]

    # remove the nan value in the thickness data
    idxval_h = jnp.where(~np.isnan(h0))[0]
    x_h = x0_h[idxval_h, None]
    y_h = y0_h[idxval_h, None]
    h = h0[idxval_h, None]
    if basal:
        s = s0[idxval_h, None]

    #%%
    # calculate the mean and range of the domain
    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2
    y_mean = jnp.mean(y)
    y_range = (y.max() - y.min()) / 2

    # calculate the mean and std of the velocity
    u_mean = jnp.mean(u)
    u_range = jnp.std(u) * 2
    v_mean = jnp.mean(v)
    v_range = jnp.std(v) * 2

    # calculate the mean and std of the thickness
    h_mean = jnp.mean(h)
    h_range = jnp.std(h) * 2

    if basal:
        s_mean = jnp.mean(s)
        s_range = jnp.std(s) * 2

    # normalize the velocity data
    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range
    u_n = (u - u_mean) / u_range
    v_n = (v - v_mean) / v_range
    if basal: 
        u_n_gl = (uraw_gl - u_mean) / u_range
        v_n_gl = (vraw_gl - v_mean) / v_range
        x_n_walls = (xdraw_walls - x_mean) / x_range 
        y_n_walls = (ydraw_walls - y_mean) / y_range
        u_n_walls = -u_mean / u_range * jnp.ones(jnp.shape(x_n_walls))
        v_n_walls = -v_mean / v_range * jnp.ones(jnp.shape(x_n_walls))

    # normalize the thickness data
    xh_n = (x_h - x_mean) / x_range
    yh_n = (y_h - y_mean) / y_range
    h_n = (h) / h_mean
    if basal:
        s_n = (s) / h_mean # choose thickness as scale for surface elevation

    # normalize the calving front position
    xct_n = (xct - x_mean) / x_range
    yct_n = (yct - y_mean) / y_range

    # group the raw data
    data_raw = [x0, y0, u0, v0, x0_h, y0_h, h0]
    if basal:
        data_raw.append(s0)
    # group the normalized data
    data_norm = [x_n, y_n, u_n, v_n, xh_n, yh_n, h_n]
    if basal:
        data_norm.append(s_n)
    # group the nan info of original data
    idxval_all = [idxval_u, idxval_h]
    # group the shape info of original data
    dsize_all = [uraw.shape, hraw.shape]

    # group the mean and range info for each variable (shape = (5,))
    data_mean=[x_mean, y_mean, u_mean, v_mean, h_mean]
    if basal:
        data_mean.append(s_mean)
    data_mean = jnp.hstack(data_mean)

    data_range=[x_range, y_range, u_range, v_range, h_range]
    if basal:
        data_range.append(s_range)
    data_range = jnp.hstack(data_range)

    if basal:
        lx0, ly0, u0, v0 = data_range[0:4]
        h0 = data_mean[4]
        # find the maximum velocity and length scale
        u0m = lax.max(u0, v0)
        l0m = lax.max(lx0, ly0)
        # calculate the scale of viscosity
        mu0 = rho * g * h0 * (l0m / u0m) / 1000.
        mu_n_gl = muraw_gl / mu0

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, idxval_all, dsize_all]

    #%% data grouping

    # group the input and output into matrix
    X_star = [jnp.hstack((x_n, y_n)), jnp.hstack((xh_n, yh_n))]
    X_ct = jnp.hstack((xct_n, yct_n))
    # sequence of output matrix column is u,v,h
    U_star = [jnp.hstack((u_n, v_n)), h_n]
    if basal:
        U_star.append(s_n)

    if basal:
        boundary_star = [u_n_gl, v_n_gl, mu_n_gl, x_n_walls, y_n_walls, u_n_walls, v_n_walls]

    if basal:
        return X_star, U_star, X_ct, nnct, data_info, boundary_star
    else:
        return X_star, U_star, X_ct, nnct, data_info
# %%
