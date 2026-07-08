'''
@author: Yongji Wang
Goal: "preprocessing_xpinns.py" normalize the observational data
for large ice shelves which are stored in different sub-regions and
organize the data into a form that is required for the PINN training

Updated by: [Agent] to include basal inversion functionality

'''

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from collections import namedtuple
from typing import NamedTuple, Tuple
from jax.typing import ArrayLike

class DataMean(NamedTuple):
    x_mean: float
    y_mean: float
    u_mean: float 
    v_mean: float 
    h_mean: float 
    s_mean: float 

class DataRange(NamedTuple):
    x_range: float 
    y_range: float
    u_range: float 
    v_range: float 
    h_range: float 
    s_range: float 
    
class DynamicScale(NamedTuple):
    l0: float
    u0: float
    mu0: float      # Scale of the viscosity (>=0) [Pa s]
    c0: float       # Scale of the friction coefficient (>=0) [Pa s/m]
    term0: float    # Scale of the viscous stress term [Pa]
    gamma_mu: float
    gamma_c: float
    rho:float = 917.
    rho_w:float = 1023.
    g:float = 9.8

class SubScaleResult(NamedTuple):
    data_mean: DataMean
    data_range: DataRange 
    dynamic_scale: DynamicScale

INTERFACE_COLLOCATION_BUFFER = 0.0


def _sample_nearest_field(x_tgt:ArrayLike, y_tgt:ArrayLike,
                          x_src:ArrayLike, y_src:ArrayLike,
                          value_src:ArrayLike) -> ArrayLike:
    x_src = jnp.asarray(x_src).flatten()
    y_src = jnp.asarray(y_src).flatten()
    value_src = jnp.asarray(value_src).flatten()
    valid = jnp.isfinite(x_src) & jnp.isfinite(y_src) & jnp.isfinite(value_src)
    x_src = x_src[valid]
    y_src = y_src[valid]
    value_src = value_src[valid]
    if x_src.size == 0:
        return jnp.full((jnp.asarray(x_tgt).size, 1), jnp.nan)

    x_tgt = jnp.asarray(x_tgt).flatten()
    y_tgt = jnp.asarray(y_tgt).flatten()
    dist_sq = (x_tgt[:, None] - x_src[None, :])**2 + (y_tgt[:, None] - y_src[None, :])**2
    idx = jnp.argmin(dist_sq, axis=1)
    return value_src[idx][:, None]

def _distance_to_interface_sq(x:ArrayLike, y:ArrayLike, x_if:ArrayLike, y_if:ArrayLike) -> ArrayLike:
    x_if = jnp.asarray(x_if).flatten()
    y_if = jnp.asarray(y_if).flatten()
    valid = jnp.isfinite(x_if) & jnp.isfinite(y_if)
    x_if = x_if[valid]
    y_if = y_if[valid]
    if x_if.size == 0:
        return jnp.inf * jnp.ones(x.shape[0])
    if x_if.size < 2:
        return jnp.min((x - x_if.T)**2 + (y - y_if.T)**2, axis=1)

    x0 = x_if[:-1]
    y0 = y_if[:-1]
    dx = x_if[1:] - x0
    dy = y_if[1:] - y0
    seg_len_sq = dx**2 + dy**2
    px = x - x0.T
    py = y - y0.T
    t = jnp.where(seg_len_sq > 0.0, (px * dx.T + py * dy.T) / seg_len_sq.T, 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    x_proj = x0.T + t * dx.T
    y_proj = y0.T + t * dy.T
    return jnp.min((x - x_proj)**2 + (y - y_proj)**2, axis=1)

def _filter_collocation_interfaces(x:ArrayLike, y:ArrayLike, interfaces, buffer_m:float) -> Tuple[ArrayLike, ArrayLike]:
    if buffer_m <= 0.0 or len(interfaces) == 0:
        return x, y
    keep = jnp.ones(x.shape[0], dtype=bool)
    for x_if, y_if in interfaces:
        keep = keep & (_distance_to_interface_sq(x, y, x_if, y_if) >= buffer_m**2)
    return x[keep], y[keep]
    
def calc_sub_scale(x:ArrayLike, y:ArrayLike, u:ArrayLike, v:ArrayLike, h:ArrayLike, s:ArrayLike, 
                   basal=False, gamma_c=0.5, gamma_mu=0.5) -> Tuple[DataMean, DataRange, DynamicScale]:

    # calculate the magnitude of each output variable for normalization later
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
    s_mean = jnp.mean(s)
    s_range = jnp.std(s) * 2 
    
    l0 = jnp.minimum(x_range, y_range)
    u0 = jnp.maximum(u_range, v_range)
    
    # Grounded and Floating ice have different dynamic scales
    scale_default = DynamicScale(0, 0, 0, 0, 0, 0, 0)
    if basal:
        mu0 = gamma_mu * scale_default.rho * scale_default.g * s_range * (l0 / u0)
        term0 = scale_default.rho * scale_default.g * h_mean * s_range / l0
        c0 = gamma_c * term0 / u0
        # term_bd = 0.0
    else:
        mu0 = (1.0 - scale_default.rho / scale_default.rho_w) * scale_default.rho * scale_default.g * h_mean * (l0 / u0)
        term0 = (1.0 - scale_default.rho/scale_default.rho_w) * scale_default.rho * scale_default.g * h_mean**2 / l0
        c0 = 1.0
        # term_bd = h_mean
    
    
    data_mean = DataMean(x_mean, y_mean, u_mean, v_mean, h_mean, s_mean)
    data_range = DataRange(x_range, y_range, u_range, v_range, h_range, s_range)
    dynamic_scale = DynamicScale(l0, u0, mu0, c0, term0, gamma_mu, gamma_c)
    
    return SubScaleResult(data_mean, data_range, dynamic_scale)


def normalize_each_legacy(data, idx, ng, basal=False):
    """Normalize older XPINN fixtures that predate surface/collocation fields.

    This keeps broad legacy tests and notebooks working without weakening the
    newer grounded/floating preprocessing contract. The branch is selected only
    by ``normalize_data`` when no basal mask is supplied and the old schema is
    detected.
    """

    xraw = data['xd'][0, idx]
    yraw = data['yd'][0, idx]
    uraw = data['ud'][0, idx]
    vraw = data['vd'][0, idx]
    xraw_h = data['xd_h'][0, idx]
    yraw_h = data['yd_h'][0, idx]
    hraw = data['hd'][0, idx]
    if basal:
        sraw = data['sd'][0, idx]

    xct = data['xct'][0, idx]
    yct = data['yct'][0, idx]
    nnct = data['nnct'][0, idx]

    if idx == 0:
        x_md = data['x_md'][0, idx]
        y_md = data['y_md'][0, idx]
    elif idx == ng-1:
        x_md = data['x_md'][0, idx-1]
        y_md = data['y_md'][0, idx-1]
    else:
        x_md = jnp.vstack([data['x_md'][0, idx-1], data['x_md'][0, idx]])
        y_md = jnp.vstack([data['y_md'][0, idx-1], data['y_md'][0, idx]])

    x0 = xraw.flatten()
    y0 = yraw.flatten()
    u0 = uraw.flatten()
    v0 = vraw.flatten()
    x0_h = xraw_h.flatten()
    y0_h = yraw_h.flatten()
    h0 = hraw.flatten()
    if basal:
        s0 = sraw.flatten()

    idxval_u = jnp.where(~np.isnan(u0))[0]
    x = x0[idxval_u, None]
    y = y0[idxval_u, None]
    u = u0[idxval_u, None]
    v = v0[idxval_u, None]

    idxval_h = jnp.where(~np.isnan(h0))[0]
    x_h = x0_h[idxval_h, None]
    y_h = y0_h[idxval_h, None]
    h = h0[idxval_h, None]
    if basal:
        s = s0[idxval_h, None]

    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2
    y_mean = jnp.mean(y)
    y_range = (y.max() - y.min()) / 2
    u_mean = jnp.mean(u)
    u_range = jnp.std(u) * 2
    v_mean = jnp.mean(v)
    v_range = jnp.std(v) * 2
    h_mean = jnp.mean(h)
    h_range = jnp.std(h) * 2
    if basal:
        s_mean = jnp.mean(s)
        s_range = jnp.std(s) * 2

    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range
    u_n = (u - u_mean) / u_range
    v_n = (v - v_mean) / v_range
    xh_n = (x_h - x_mean) / x_range
    yh_n = (y_h - y_mean) / y_range
    h_n = h / h_mean
    if basal:
        s_n = s / h_mean

    xct_n = (xct - x_mean) / x_range
    yct_n = (yct - y_mean) / y_range
    xmd_n = (x_md - x_mean) / x_range
    ymd_n = (y_md - y_mean) / y_range

    data_raw = [x0, y0, u0, v0, x0_h, y0_h, h0]
    data_norm = [x_n, y_n, u_n, v_n, xh_n, yh_n, h_n]
    data_mean = jnp.hstack([x_mean, y_mean, u_mean, v_mean, h_mean])
    data_range = jnp.hstack([x_range, y_range, u_range, v_range, h_range])
    if basal:
        data_raw.append(s0)
        data_norm.append(s_n)
        data_mean = jnp.hstack([data_mean, s_mean])
        data_range = jnp.hstack([data_range, s_range])

    data_info = [data_mean, data_range, data_norm, data_raw,
                 [idxval_u, idxval_h], [uraw.shape, hraw.shape]]
    X_star = [jnp.hstack((x_n, y_n)), jnp.hstack((xh_n, yh_n))]
    U_star = [jnp.hstack((u_n, v_n)), h_n]
    if basal:
        U_star.append(s_n)

    X_ct = jnp.hstack((xct_n, yct_n))
    X_md = jnp.hstack((xmd_n, ymd_n))
    if basal:
        return X_star, U_star, X_ct, nnct, data_info, X_md, None
    return X_star, U_star, X_ct, nnct, data_info, X_md
    
# function to load the data for each sub-regions
def normalize_each(data, idx, ng, basal=False, use_regression=False, forward_mode=False,
                   basal_mask_all=None,
                   grounded_only_interface_mu_ct:bool=False,
                   interface_mu_source:str='floating'):

    '''
    :param data: data for all sub-regions
    :param idx: idx for the sub-region
    :return X_smp, U_smp, X_ct, n_ct, data_info
    '''
    # extract the velocity data
    xraw = data['xd'][0, idx]  # unit [m] position of observation data points
    yraw = data['yd'][0, idx]  # unit [m] position of observation data points
    uraw = data['ud'][0, idx]  # unit [m/year] ice velocity
    vraw = data['vd'][0, idx]  # unit [m/year] ice velocity
    if use_regression:
        muraw = data['mud'][0, idx]
        Craw  = data['alpha2d'][0, idx]

    # extract the thickness data (may have different position)
    xraw_h = data['xd_h'][0, idx]  # unit [m] position
    yraw_h = data['yd_h'][0, idx]  # unit [m] position
    hraw = data['hd'][0, idx]  # unit [m] ice thickness
    sraw = data['sd'][0, idx]
    
    # [NOTE]: 3/13/2026 added collocation points as a requirement for incoming datasets 
    xcolraw = data['xcol'][0, idx] # unit [m] position of collocation points
    ycolraw = data['ycol'][0, idx] # unit [m] position of collocation points
    if forward_mode: 
        xdirraw = data['xdir'][0, idx] # Dirichlet boundary points x [m] 
        ydirraw = data['ydir'][0, idx] # Dirichlet boundary points y [m] 
        udirraw = data['udir'][0, idx]
        vdirraw = data['vdir'][0, idx]

    # extract the position of the calving front (right side of the domain)
    xct = data['xct'][0, idx]
    yct = data['yct'][0, idx]
    nnct = data['nnct'][0, idx]

    # extract the position of interface between two nearby sub-regions
    if idx == 0:
        x_md = data['x_md'][0, idx]
        y_md = data['y_md'][0, idx]
    elif idx == ng-1:
        x_md = data['x_md'][0, idx-1]
        y_md = data['y_md'][0, idx-1]
    else:
        x_md = jnp.vstack([data['x_md'][0, idx-1], data['x_md'][0, idx]])
        y_md = jnp.vstack([data['y_md'][0, idx-1], data['y_md'][0, idx]])

    bd_mu_raw = None
    interface_idx = None
    if basal and grounded_only_interface_mu_ct and use_regression:
        if interface_mu_source != 'floating':
            raise ValueError('[xpinn :: normalize_each] Only interface_mu_source="floating" is supported.')
        if basal_mask_all is None:
            raise ValueError('[xpinn :: normalize_each] basal_mask_all is required for grounded_only_interface_mu_ct.')
        source_idx = None
        if idx > 0 and not basal_mask_all[idx-1]:
            source_idx = idx - 1
        elif idx < ng - 1 and not basal_mask_all[idx+1]:
            source_idx = idx + 1
        if source_idx is None:
            raise ValueError('[xpinn :: normalize_each] grounded_only_interface_mu_ct requires a grounded region adjacent to a floating region.')
        interface_idx = min(idx, source_idx)
        x_if = jnp.asarray(data['x_md'][0, interface_idx]).flatten()
        y_if = jnp.asarray(data['y_md'][0, interface_idx]).flatten()
        x_src = jnp.asarray(data['xd'][0, source_idx]).flatten()
        y_src = jnp.asarray(data['yd'][0, source_idx]).flatten()
        mu_src = jnp.asarray(data['mud'][0, source_idx]).flatten()
        bd_mu_raw = _sample_nearest_field(x_if, y_if, x_src, y_src, mu_src)

    # flatten the velocity data into 1d array
    x0 = xraw.flatten()
    y0 = yraw.flatten()
    u0 = uraw.flatten()
    v0 = vraw.flatten()
    
    if use_regression:
        mu0 = muraw.flatten()
        C0  = Craw.flatten()
    
    xcol0 = xcolraw.flatten()
    ycol0 = ycolraw.flatten()
    interfaces = []
    if idx > 0:
        interfaces.append((data['x_md'][0, idx-1], data['y_md'][0, idx-1]))
    if idx < ng - 1:
        interfaces.append((data['x_md'][0, idx], data['y_md'][0, idx]))
    xcol0, ycol0 = _filter_collocation_interfaces(
        xcol0[:, None], ycol0[:, None], interfaces, INTERFACE_COLLOCATION_BUFFER)
    xcol0 = xcol0.flatten()
    ycol0 = ycol0.flatten()
    # if use_regression:
    #     xdir0 = xdirraw.flatten()
    #     ydir0 = ydirraw.flatten()
    #     udir0 = udirraw.flatten()
    #     vdir0 = vdirraw.flatten()

    # flatten the thickness data into 1d array
    x0_h = xraw_h.flatten()
    y0_h = yraw_h.flatten()
    h0 = hraw.flatten()
    s0 = sraw.flatten()

    # remove the nan value in the velocity data
    idxval_u = jnp.where(~np.isnan(u0))[0]
    x = x0[idxval_u, None]
    y = y0[idxval_u, None]
    u = u0[idxval_u, None]
    v = v0[idxval_u, None]
    if use_regression:
        mu = mu0[idxval_u, None]
        C  = C0[idxval_u, None]
    
    xcol = xcol0[:, None]
    ycol = ycol0[:, None]

    # remove the nan value in the thickness data
    idxval_h = jnp.where(~np.isnan(h0))[0]
    x_h = x0_h[idxval_h, None]
    y_h = y0_h[idxval_h, None]
    h = h0[idxval_h, None]
    s = s0[idxval_h, None]

    subscale_result = calc_sub_scale(x, y, u, v, h, s, basal=basal)
    data_mean, data_range, dynamic_scale = subscale_result

    # normalize the velocity data
    x_n = (x - data_mean.x_mean) / data_range.x_range
    y_n = (y - data_mean.y_mean) / data_range.y_range
    u_n = (u - data_mean.u_mean) / data_range.u_range
    v_n = (v - data_mean.v_mean) / data_range.v_range
    if use_regression:
        mu_n = mu / dynamic_scale.mu0 
        if basal:
            C_n = C / dynamic_scale.c0
        else:
            C_n = jnp.zeros_like(mu_n)
    
    xcol_n = (xcol - data_mean.x_mean) / data_range.x_range
    ycol_n = (ycol - data_mean.y_mean) / data_range.y_range 

    # normalize the thickness data
    xh_n = (x_h - data_mean.x_mean) / data_range.x_range
    yh_n = (y_h - data_mean.y_mean) / data_range.y_range
    h_n = (h) / data_mean.h_mean
    s_n = (s - data_mean.s_mean) / data_range.s_range

    if basal and grounded_only_interface_mu_ct and bd_mu_raw is not None:
        xct = data['x_md'][0, interface_idx]
        yct = data['y_md'][0, interface_idx]
    xct_n = (xct - data_mean.x_mean) / data_range.x_range
    yct_n = (yct - data_mean.y_mean) / data_range.y_range

    # normalize the interface position between subregions
    xmd_n = (x_md - data_mean.x_mean) / data_range.x_range
    ymd_n = (y_md - data_mean.y_mean) / data_range.y_range

    # group the raw data
    data_raw = [x0, y0, u0, v0, x0_h, y0_h, h0, s0, xcol0, ycol0, x_md, y_md]
    if use_regression:
        data_raw.append(muraw)
        data_raw.append(Craw)
    
    # group the normalized data
    data_norm = [x_n, y_n, u_n, v_n, xh_n, yh_n, h_n, s_n, xcol_n, ycol_n, xmd_n, ymd_n]
    if use_regression:
        data_norm.append(mu_n)
        data_norm.append(C_n)

    # group the nan info of original data
    idxval_all = [idxval_u, idxval_h]
    # group the shape info of original data
    dsize_all = [uraw.shape, hraw.shape]

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, idxval_all, dsize_all, subscale_result]

    # group the input and output into matrix
    X_star = [jnp.hstack((x_n, y_n)), jnp.hstack((xh_n, yh_n)), jnp.hstack((xcol_n, ycol_n))]
    X_ct = jnp.hstack((xct_n, yct_n))
    X_md = jnp.hstack((xmd_n, ymd_n))

    # sequence of output matrix column is u,v,h (and s if basal)
    U_star = [jnp.hstack((u_n, v_n)), h_n, s_n]
    if use_regression:
        U_star.append(mu_n)
        U_star.append(C_n)
    
    if basal:
        boundary_star = None if bd_mu_raw is None else [bd_mu_raw / dynamic_scale.mu0]
        return X_star, U_star, X_ct, nnct, data_info, X_md, boundary_star
    else:
        return X_star, U_star, X_ct, nnct, data_info, X_md

# function to load the data for all sub-regions
def normalize_data(data, basal_mask=None, use_regression=False,
                   grounded_only_interface_mu_ct:bool=False,
                   interface_mu_source:str='floating'):
    # count the number of sub-regions
    ng = len(data['xd'][0])
    
    use_legacy_all_floating = basal_mask is None and ('sd' not in data or 'xcol' not in data)
    if basal_mask is None:
        basal_mask = [False] * ng
        
    # create an index list for different sub-regions
    idxgall = jnp.arange(ng).tolist()
    # load the data for each sub-regions
    if use_legacy_all_floating:
        data_all = tree_map(lambda x, b: normalize_each_legacy(data, x, ng, basal=b), idxgall, basal_mask)
    else:
        data_all = tree_map(
            lambda x, b: normalize_each(
                data, x, ng, basal=b, use_regression=use_regression,
                basal_mask_all=basal_mask,
                grounded_only_interface_mu_ct=grounded_only_interface_mu_ct,
                interface_mu_source=interface_mu_source),
            idxgall, basal_mask)

    # exact the postion matrix of velocity data for entire ice shelves
    Xe = data['Xe']
    Ye = data['Ye']
    # exact the postion matrix of thickness data for entire ice shelves
    Xe_h = data['Xe_h']
    Ye_h = data['Ye_h']
    # group the entire position matrix
    posi_all = [Xe, Ye, Xe_h, Ye_h]

    # obtain the location of each subregion in the entire ice-shelf matrix
    idxcrop = data['idxcrop']      # for velocity
    idxcrop_h = data['idxcrop_h']  # for thickness
    # convert the idxcrop to a array (for simple calculation later)
    idxcrop = jnp.array(idxcrop.tolist()).reshape(ng, 4)
    idxcrop_h = jnp.array(idxcrop_h.tolist()).reshape(ng, 4)

    # group the idxcrop
    idxcrop_all = [idxcrop, idxcrop_h]

    return data_all, idxgall, posi_all, idxcrop_all
