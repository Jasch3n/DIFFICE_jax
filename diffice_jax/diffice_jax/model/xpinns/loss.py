import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax
import jax.debug as jdb


# define the mean squared error
def ms_error(diff):
    return jnp.sum(jnp.square(diff), axis=0) / jnp.maximum(diff.shape[0], 1.0)


# take the nth power root with original sign
def nthrt(x, n):
    return jnp.sign(x) * jnp.abs(x) ** (1/n)


def sub_scale(scale, basal=False):
    # define the global parameter
    rho = 917
    rho_w = 1023
    g = 9.8
    gd = g * (1 - rho / rho_w)  # reduced gravitational acceleration
    # load the scale information
    dmean, drange = scale
    lx0, ly0, u0, v0 = drange[0:4]
    um, vm, h0 = dmean[2:5]
    
    u0m = lax.max(u0, v0)
    l0m = lax.min(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    # Use full gravity for grounded ice (matching PINN behavior)
    g_eff = g if basal else gd
    mu0 = rho * g_eff * h0 * (l0m / u0m)
    du0 = u0m / l0m
    dh0 = h0 / l0m

    # [TODO]: Figure out what the right expression for term 0 is ...
    term0 = h0**2 / l0m
    return u0, v0, h0, mu0, du0, dh0, term0, um/u0, vm/v0


def u_mag(u):
    return jnp.sqrt(jnp.sum(jnp.square(u), 1))


#%% loss for inferring isotropic viscosity

def loss_iso_create(solNN, eqn_all, scale, idxgall, lw, basal_mask=None, gamma_eq=None):
    ''' a function factory to create the loss function for isotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :param gamma_eq: optional equation weight override for MSNN higher stages.
                     If provided, loss = (1-gamma_eq)*loss_data + gamma_eq*loss_eqn.
                     If None, uses the standard lw-based weighting.
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all
    
    # create default basal mask if not provided (all floating)
    if basal_mask is None:
        ng = len(idxgall)
        basal_mask = [False] * ng

    # obtain the viscosity and strain rate scale in each sub-region
    # Pass basal flag per region so grounded regions use rho*g (not rho*gd)
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x], basal=basal_mask[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        is_basal = basal_mask[idx]

        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)

        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]

        if is_basal:
            s_smp = data['smp'][4][idx]

        # load the position of collocation points
        x_col = data['col'][0][idx]

        # load boundary data (only for floating subregions)
        if not is_basal:
            x_bd = data['bd'][0][idx]
            nn_bd = data['bd'][1][idx]

        # network predictions
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        if is_basal:
            s_pred = net(xh_smp)[:, 3:4]

        # ================= CALCULATE LOSSES FOR SUBREGION =================
        # equation residual
        f_pred = gov_eqn(net, x_col, scale[idx], basal=is_basal)[0]

        # data errors
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)

        if is_basal:
            #[TODO]: Figure out how to properly scale basal velocity data loss
            # Reason: Velocity of grounded ice is much smaller 
            # data_u_err *= 100.0 

            data_s_err = ms_error(s_pred - s_smp)

            # data_err: [u_err(2), h_err(1)] scaled, plus s_err and log_u_err
            # data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]
            # s_err_weighted = data_s_err * h0[idx]
            data_err = jnp.hstack((data_u_err, data_h_err))
            s_err_weighted = data_s_err
            
            data_err_all = jnp.hstack([data_err, s_err_weighted])  # (4,)

            # equation error
            # [TODO]: Figure out the right weight balance for equation error 
            # Reason: term0 is prop to 1/l0m, which is about 1000 larger for a pinning point than for ice shelves
            # eqn_err = ms_error(f_pred) * term0[idx] / 1e3  # (2,)
            eqn_err = ms_error(f_pred) # (2,)

            # no boundary conditions for grounded regions
            bd_err_vec = jnp.array([0.0, 0.0])  # (2,)

            err_all = jnp.hstack([data_err_all, eqn_err, bd_err_vec])

        else:
            # floating case
            # data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]  # (3,)
            data_err = jnp.hstack((data_u_err, data_h_err))  # (3,)
            # pad s_err and log_u_err slots with 0 (not applicable for floating)
            data_err_all = jnp.hstack([data_err, 0.0])  # (4,)

            # equation error
            # eqn_err = ms_error(f_pred) * term0[idx]  # (2,)
            eqn_err = ms_error(f_pred)  # (2,)

            # calving front boundary error
            f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]
            # bd_err = ms_error(f_bd) * h0[idx]  # (2,)
            bd_err = ms_error(f_bd)  # (2,)

            err_all = jnp.hstack([data_err_all, eqn_err, bd_err])

        # err_all size: 4 (data) + 2 (eqn) + 2 (bd) = 8
        return err_all

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # jdb.print('......Doing matching loss calculation for regionIdx={x}', x=idx)
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        
        # Need to pass basal flag to gov_eqn if we use it in matching (C2)
        # gov_eqn signature updated to accept basal=...
        # But here we invoke it.
        # Check basal status of idx and idx+1
        is_basal_1 = basal_mask[idx]
        is_basal_2 = basal_mask[idx+1]
        
        fgovterm = lambda x, id, b: gov_eqn(lambda z: net(z, id), x, scale[id], basal=b)[1]
        
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        u0m = lax.max(u0[idx], u0[idx+1])
        v0m = lax.max(v0[idx], v0[idx+1])
        h0m = lax.max(h0[idx], h0[idx+1])
        mu0m = lax.max(mu0[idx], mu0[idx+1])

        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx] / u0m
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx] / v0m
        h_md1 = (U_md1[:, 2:3]) * h0[idx] / h0m
        mu_idx_1 = 4 if basal_mask[idx] else 3
        mu_md1 = (U_md1[:, mu_idx_1:mu_idx_1+1]) * mu0[idx] / mu0m
        # jdb.print("mean u_md1: {x}", x=jnp.mean(u_md1))
        # jdb.print("mean v_md1: {x}", x=jnp.mean(v_md1))
        # jdb.print("mean h_md1: {x}", x=jnp.mean(h_md1))
        # jdb.print("mu_md1: {x}", x=mu_md1)
        
        # vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2 * jnp.log(mu_md1)])
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, mu_md1])
        
        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx + 1)
        u_md2 = (U_md2[:, 0:1] + um[idx + 1]) * u0[idx + 1] / u0m
        v_md2 = (U_md2[:, 1:2] + vm[idx + 1]) * v0[idx + 1] / v0m
        h_md2 = (U_md2[:, 2:3]) * h0[idx + 1] / h0m
        mu_idx_2 = 4 if basal_mask[idx+1] else 3
        mu_md2 = (U_md2[:, mu_idx_2:mu_idx_2+1]) * mu0[idx + 1] / mu0m
        # jdb.print("mean u_md2: {x}", x=jnp.mean(u_md2))
        # jdb.print("mean v_md2: {x}", x=jnp.mean(v_md2))
        # jdb.print("mean h_md2: {x}", x=jnp.mean(h_md2))
        # jdb.print("mu_md2: {x}", x=mu_md2)
        # jdb.print("mean mu_md2: {x}", x=jnp.mean(2 * jnp.log(mu_md2)))
        
        # vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2 * jnp.log(mu_md1)])
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, mu_md1])
        
        # vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2 * jnp.log(mu_md2)])
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, mu_md2])
        
        # [NEW]: Apply one-way constraint for all variables at the grounding line
        if is_basal_1 and not is_basal_2:
            # Region 1 is grounded, Region 2 is floating
            # We want Region 1 to conform to Region 2 without altering Region 2
            vars_md2 = lax.stop_gradient(vars_md2)
            
        elif not is_basal_1 and is_basal_2:
            # Region 1 is floating, Region 2 is grounded
            # We want Region 2 to conform to Region 1 without altering Region 1
            vars_md1 = lax.stop_gradient(vars_md1)
        
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)
        # jdb.print("mean match_c0_err: {x}", x=jnp.where(jnp.isnan(vars_md1 - vars_md2), 0, 1))
        # jdb.print("mean match_c0_err: {x}", x=jnp.mean(match_c0_err))

        """C1 stitching condition at the boundary"""
        du0m = lax.max(du0[idx], du0[idx+1])
        dh0m = lax.max(dh0[idx], dh0[idx+1])

        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx] / du0m
        dh_md1 = dU_md1[:, 4:6] * dh0[idx] / dh0m
        if is_basal_1 != is_basal_2:
            dvars_md1 = jnp.hstack([duv_md1])
        else:
            dvars_md1 = jnp.hstack([duv_md1, dh_md1])

        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx + 1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx + 1] / du0m
        dh_md2 = dU_md2[:, 4:6] * dh0[idx + 1] / dh0m
        if is_basal_1 != is_basal_2:
            dvars_md2 = jnp.hstack([duv_md2])
        else:
            dvars_md2 = jnp.hstack([duv_md2, dh_md2])
            
        # [NEW]: Apply one-way constraint for all C1 variables at the grounding line 
        # if is_basal_1 and not is_basal_2:
        #     dvars_md2 = lax.stop_gradient(dvars_md2)
        # elif not is_basal_1 and is_basal_2:
        #     dvars_md1 = lax.stop_gradient(dvars_md1)
            
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        # gov_eqn returns (f_eqn, val_term). we want val_term.
        # match terms depending on interface type
        if is_basal_1 != is_basal_2:
            # C2 continuity is not satisfied at grounding lines
            match_c2_err = 0.0
        else: 
            term_md1 = fgovterm(x_md[:, 0:2], idx, is_basal_1)[:, 0:-1] * term0[idx]
            term_md2 = fgovterm(x_md[:, 2:4], idx + 1, is_basal_2)[:, 0:-1] * term0[idx + 1]
            if basal_mask is not None:
                # calculate equation residue in sub-region 1 at the interface
                # gov_eqn returns (f_eqn, val_term). we want val_term.
                # Same-type interface with basal regions: match first 6 terms
                match_c2_err = ms_error(nthrt(term_md1[:, 0:6], 2) - nthrt(term_md2[:, 0:6], 2))
            else:
                # Ice shelf interface: match all terms 
                match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitched conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        # jdb.print("mc0_err: {x} | mc1_err: {y} | mc2_err: {z}", x=mc0_err, y=mc1_err, z=mc2_err)
        # jdb.print("mc1_err: {x}", x=mc1_err)
        # jdb.print("mc2_err: {x}", x=mc2_err)
        # [NOTE]: Turn on/off the different continuity terms as needed 
        # match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.2])
        match_err = jnp.hstack([mc0_err, mc1_err*0.7, mc2_err*0.2])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        reg_err_list = jnp.array(tree_map(lambda x: loss_sub(params, data, x), idxgall))

        reg_err = jnp.mean(reg_err_list, axis=0)
        # calculate the error at the matching boundary
        match_err_list = jnp.array(tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1]))
        match_err = jnp.mean(match_err_list, axis=0)
        # jdb.print('===============================================================================\n'
        #           '\t\tfloat errs: u={f1:.3e}  | v={f2:.3e}  | h={f3:.3e} | s=N/A\n'
        #           '\t\t            e1={f4:.3e} | e2={f5:.3e} | bd1={bd1:.3e} | bd2={bd2:.3e}\n'
        #           '\t\tgrnd  errs: u={g1:.3e}  | v={g2:.3e}  | h={g3:.3e} | s={g4:.3e}\n'
        #           '\t\t            e1={g5:.3e} | e2={g6:.3e} | bd1={bd3:.3e} | bd2={bd4:.3e}\n'
        #           '\t\tmatch errs: mc0={mc0:.3e} | mc1={mc1:.3e} | mc2={mc2:.3e}',
        #           f1=reg_err_list[0,0], f2=reg_err_list[0,1], f3=reg_err_list[0,2], f4=reg_err_list[0,4], f5=reg_err_list[0,5],
        #           bd1=reg_err_list[0,6], bd2=reg_err_list[0,7],
        #           g1=reg_err_list[1,0], g2=reg_err_list[1,1], g3=reg_err_list[1,2], g4=reg_err_list[1,3], g5=reg_err_list[1,4], g6=reg_err_list[1,5],
        #           bd3=reg_err_list[1,6], bd4=reg_err_list[1,7],
        #           mc0=match_err[0], mc1=match_err[1], mc2=match_err[2])

        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        # data_w (u, v, h, s) -> 1, 1, 0.6, 0.6
        data_w = jnp.array([1., 1., 0.6, 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        md_w = jnp.array([1., 1., 1.])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, md_w])

        # calculate the overall data loss and equation loss
        loss_each = err_all * wgh_all
        loss_data = jnp.sum(loss_each[0:4])
        loss_eqn = jnp.sum(loss_each[4:6])
        loss_bd = jnp.sum(loss_each[6:8])
        loss_md = jnp.sum(loss_each[8:])
        # jdb.print("loss_data: {x}", x=loss_data)
        # jdb.print("loss_eqn: {x}", x=loss_eqn)
        # jdb.print("loss_bd: {x}", x=loss_bd)
        # jdb.print("loss_md: {x}", x=loss_md)
        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # load the (possibly mutated) loss weights
        _lw = loss_fun.lw
        # calculate the total loss
        _gamma_eq = loss_fun.gamma_eq
        if _gamma_eq is not None:
            # MSNN mode: use gamma_eq to balance data vs equation loss
            loss = ((1.0 - _gamma_eq) * (loss_data + loss_bd + loss_md)
                    + _gamma_eq * loss_eqn)
        else:
            # Standard mode
            loss = (_lw[0] * loss_data + _lw[1] * loss_eqn + _lw[2] * loss_bd + _lw[3] * loss_md)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md]), err_all])
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0
    # store loss weights as mutable attribute (can be updated between stages)
    loss_fun.lw = jnp.array(lw)
    # store MSNN gamma_eq (None = standard mode)
    loss_fun.gamma_eq = gamma_eq

    return loss_fun



#%% loss for inferring anisotropic viscosity

def loss_aniso_create(solNN, eqn_all, scale, idxgall, lw):
    ''' a function factory to create the loss function for anisotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)
        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]

        # load the position and weight of collocation points
        x_col = data['col'][0][idx]
        x_bd = data['bd'][0][idx]
        nn_bd = data['bd'][1][idx]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]

        # calculate the residue of equation
        f_pred = gov_eqn(net, x_col, scale[idx])[0]
        f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]

        # calculate the mean squared error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]
        # calculate the mean squared error of equation
        eqn_err = ms_error(f_pred) * term0[idx]
        bd_err = ms_error(f_bd) * h0[idx]
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2) * mu0[idx]

        # group all the error for output
        err_all = jnp.hstack([data_err, eqn_err, bd_err, sp_err])
        return err_all, f_pred

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        fgovterm = lambda x, id: gov_eqn(lambda x: net(x, id), x, scale[id])[1]
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx]
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx]
        h_md1 = (U_md1[:, 2:3]) * h0[idx]
        mu_md1 = (U_md1[:, 3:5]) * mu0[idx]     # include both mu and eta
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2*jnp.log(mu_md1)])
        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx+1)
        u_md2 = (U_md2[:, 0:1] + um[idx+1]) * u0[idx+1]
        v_md2 = (U_md2[:, 1:2] + vm[idx+1]) * v0[idx+1]
        h_md2 = (U_md2[:, 2:3]) * h0[idx+1]
        mu_md2 = (U_md2[:, 3:5]) * mu0[idx+1]   # include both mu and eta
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2*jnp.log(mu_md2)])
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)

        """C1 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx]
        dh_md1 = dU_md1[:, 4:6] * dh0[idx]
        dvars_md1 = jnp.hstack([duv_md1, dh_md1])
        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx+1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx+1]
        dh_md2 = dU_md2[:, 4:6] * dh0[idx+1]
        dvars_md2 = jnp.hstack([duv_md2, dh_md2])
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        term_md1 = fgovterm(x_md[:, 0:2], idx)[:, 0:-1] * term0[idx]
        # calculate equation residue in sub-region 2 at the interface
        term_md2 = fgovterm(x_md[:, 2:4], idx+1)[:, 0:-1] * term0[idx+1]
        # calculate the c2 error
        match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitching conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.5])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        # Unpack the tuple return from loss_sub
        sub_results = [loss_sub(params, data, x) for x in idxgall]
        reg_err_list = [res[0] for res in sub_results]
        residuals_list = [res[1] for res in sub_results]

        reg_err = jnp.mean(jnp.array(reg_err_list), axis=0)
        # calculate the error at the matching boundary
        match_err_list = tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1])
        match_err = jnp.mean(jnp.array(match_err_list), axis=0)
        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        data_w = jnp.array([1., 1., 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        sp_w = jnp.array([1.])
        md_w = jnp.ones(match_err.shape[0])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, sp_w, md_w])

        # modify the contribution of each loss term by their weights
        loss_each = err_all * wgh_all
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(loss_each[0:3])
        loss_eqn = jnp.sum(loss_each[3:5])
        loss_bd = jnp.sum(loss_each[5:7])
        loss_sp = jnp.sum(loss_each[7:8])
        loss_md = jnp.sum(loss_each[8:])

        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # load the weight for the regularization loss
        wsp = loss_fun.wsp
        # calculate the total loss
        loss = (lw[0] * loss_data + lw[1] * loss_eqn + lw[2] * loss_bd + lw[3] * loss_md + wsp * loss_sp)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = [jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md, loss_sp]), err_all]), residuals_list]
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0
    loss_fun.wsp = lw[4]
    return loss_fun