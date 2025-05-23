import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax

def dataArrange(var, idxval, dsize):
    nanmat = jnp.empty(dsize)
    nanmat = nanmat.at[:].set(jnp.nan)
    var_1d = nanmat.flatten()[:, None]
    var_1d = var_1d.at[idxval].set(var)
    var_2d = jnp.reshape(var_1d, dsize)
    return var_2d

def extract_scale(scale_info, basal=False):
    # define the global parameter
    rho = 917
    rho_w = 1030
    g = 9.8
    gd = g * (1 - rho / rho_w)  # gravitational acceleration
    # load the scale information
    dmean, drange = scale_info
    lx0, ly0, u0, v0 = drange[0:4]
    lxm, lym, um, vm = dmean[0:4]
    h0 = dmean[4]
    # find the maximum velocity and length scale
    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    if basal:
        mu0 = rho * g * h0 * (l0m / u0m)
        c0 = h0 * mu0 / (l0m ** 2)
    else:
        mu0 = rho * gd * h0 * (l0m / u0m)
        c0 = jnp.nan
    str0 = u0m/l0m
    term0 = rho * gd * h0 ** 2 / l0m
    # group characteristic scales for all different variables
    scale = dict(lx0=lx0, ly0=ly0, u0=u0, v0=v0, h0=h0,
                 lxm=lxm, lym=lym, um=um, vm=vm,
                 mu0=mu0, str0=str0, term0=term0, c0=c0)
    return scale


def predict(func_all, data_all, aniso=False, basal=False):
    # obtain the normalized dataset
    x_star, y_star, u_star, v_star, xh_star, yh_star, h_star = data_all[4][2]
    # set the output position based on the original velocity data
    x_pred = jnp.hstack([x_star, y_star])
    # set the output position based on the original thickness data
    xh_pred = jnp.hstack([xh_star, yh_star])

    # obtain the non-nan index of the original dataset
    idxval, idxval_h = data_all[4][-2]
    # obtain the 2D shape of the original dataset
    dsize, dsize_h = data_all[4][-1]
    # extract the scale for different variables
    scale = data_all[4][0:2]
    varscl = extract_scale(scale, basal=basal)

    # extract the function of solution and equation residue
    [f_u, f_gu, gov_eqn] = func_all
    if basal:
        ocean_mask = data_all[-1]
    else:
        ocean_mask = None
    print("predict: ", jnp.shape(ocean_mask))
    f_eqn = lambda x: gov_eqn(f_u, x, scale, basal=basal)

    # calculate the network output at the original velocity-data positions
    uvhm = f_u(x_pred)
    # calculate the network output at the original thickness-data positions
    h2 = f_u(xh_pred)[:, 2:3]

    # set the partition number
    nsp = 4
    # separate input into different partition to avoid GPU memory limit
    x_psp = jnp.array_split(x_pred, nsp)
    ocean_mask_psp = jnp.array_split(ocean_mask, nsp)
    idxsp = jnp.arange(nsp).tolist()
    # calculate the derivative of network output at the velocity-data positions
    du_list = tree_map(lambda x: f_gu(x_psp[x]), idxsp)
    # calculate the associated equation residue of the trained network
    eqnterm_list = tree_map(lambda x: f_eqn(x_psp[x]), idxsp)
    if basal:
        eqn_list = tree_map(lambda x: eqnterm_list[x][0], idxsp)
        eqn_grounded_list = tree_map(lambda x: eqnterm_list[x][1], idxsp)
        term_list = tree_map(lambda x: eqnterm_list[x][2], idxsp)
    else:
        eqn_list = tree_map(lambda x: eqnterm_list[x][0], idxsp)
        term_list = tree_map(lambda x: eqnterm_list[x][1], idxsp)
    # combine the sub-group list into a long array
    duvh = jnp.vstack(du_list)
    eqn = jnp.vstack(eqn_list)
    if basal:
        eqn_grounded = jnp.vstack(eqn_grounded_list)
    term = jnp.vstack(term_list)

    # convert to 2D original velocity dataset
    x = dataArrange(x_star, idxval, dsize) * varscl['lx0'] + varscl['lxm']
    y = dataArrange(y_star, idxval, dsize) * varscl['ly0'] + varscl['lym']
    u_data = dataArrange(u_star, idxval, dsize) * varscl['u0'] + varscl['um']
    v_data = dataArrange(v_star, idxval, dsize) * varscl['v0'] + varscl['vm']

    # convert to 2D original thickness dataset
    x_h = dataArrange(xh_star, idxval_h, dsize_h) * varscl['lx0'] + varscl['lxm']
    y_h = dataArrange(yh_star, idxval_h, dsize_h) * varscl['ly0'] + varscl['lym']
    h_data = dataArrange(h_star, idxval_h, dsize_h) * varscl['h0']

    # convert to 2D NN prediction
    u_p = dataArrange(uvhm[:, 0:1], idxval, dsize) * varscl['u0'] + varscl['um']
    v_p = dataArrange(uvhm[:, 1:2], idxval, dsize) * varscl['v0'] + varscl['vm']
    h_p = dataArrange(uvhm[:, 2:3], idxval, dsize) * varscl['h0']
    h_p2 = dataArrange(h2, idxval_h, dsize_h) * varscl['h0']
    mu_p = dataArrange(uvhm[:, 3:4], idxval, dsize) * varscl['mu0']
    if aniso:
        eta_p = dataArrange(uvhm[:, 4:5], idxval, dsize) * varscl['mu0']
    if basal:
        c = dataArrange(uvhm[:, 4:5], idxval, dsize) * varscl['c0']

    # convert to 2D derivative of prediction
    ux_p = dataArrange(duvh[:, 0:1], idxval, dsize) * varscl['u0']/varscl['lx0']
    uy_p = dataArrange(duvh[:, 1:2], idxval, dsize) * varscl['u0']/varscl['ly0']
    vx_p = dataArrange(duvh[:, 2:3], idxval, dsize) * varscl['v0']/varscl['lx0']
    vy_p = dataArrange(duvh[:, 3:4], idxval, dsize) * varscl['v0']/varscl['ly0']
    hx_p = dataArrange(duvh[:, 4:5], idxval, dsize) * varscl['h0']/varscl['lx0']
    hy_p = dataArrange(duvh[:, 5:6], idxval, dsize) * varscl['h0']/varscl['ly0']

    # convert to 2D equation residue
    e1 = dataArrange(eqn[:, 0:1], idxval, dsize) * varscl['term0']
    e2 = dataArrange(eqn[:, 1:2], idxval, dsize) * varscl['term0']
    if basal: 
        e1_grounded = dataArrange(eqn_grounded[:, 0:1], idxval, dsize) * varscl['term0']
        e2_grounded = dataArrange(eqn_grounded[:, 1:2], idxval, dsize) * varscl['term0']
        e1 = ocean_mask*e1 + (1-ocean_mask)*e1_grounded 
        e2 = ocean_mask*e2 + (1-ocean_mask)*e2_grounded

    # convert to 2D equation term value
    e11 = dataArrange(term[:, 0:1], idxval, dsize) * varscl['term0']
    e12 = dataArrange(term[:, 1:2], idxval, dsize) * varscl['term0']
    e13 = dataArrange(term[:, 2:3], idxval, dsize) * varscl['term0']
    e21 = dataArrange(term[:, 3:4], idxval, dsize) * varscl['term0']
    e22 = dataArrange(term[:, 4:5], idxval, dsize) * varscl['term0']
    e23 = dataArrange(term[:, 5:6], idxval, dsize) * varscl['term0']
    e14 = None 
    e24 = None 
    if basal:
        e14 = dataArrange(term[:, 7:8], idxval, dsize) * varscl['term0']
        e24 = dataArrange(term[:, 8:9], idxval, dsize) * varscl['term0']
        e13_grounded = dataArrange(term[:, 9:10], idxval, dsize) * varscl['term0']
        e23_grounded = dataArrange(term[:, 10:11], idxval, dsize) * varscl['term0']
        e13 = ocean_mask*e13 + (1-ocean_mask)*e13_grounded 
        e23 = ocean_mask*e23 + (1-ocean_mask)*e23_grounded 
    strate = dataArrange(term[:, -1:], idxval, dsize) * varscl['str0']

    # group all the variables
    results = {"x": x, "y": y, "u_g": u_data, "v_g": v_data,
               "x_h": x_h, "y_h": y_h, "h_g": h_data,
               "u": u_p, "v": v_p, "h": h_p, "h2": h_p2,
               "u_x": ux_p, "u_y": uy_p, "v_x": vx_p, "v_y": vy_p,
               "h_x": hx_p, "h_y": hy_p, "str": strate, "mu": mu_p,
               "e11": e11, "e12": e12, "e13": e13,
               "e21": e21, "e22": e22, "e23": e23,
               "e1": e1, "e2": e2, "scale": varscl, "ocean_mask":ocean_mask}
    if aniso:
        results['eta'] = eta_p
    if basal:
        results['c'] = c
        results['e14']=e14 
        results['e24']=e24

    return results