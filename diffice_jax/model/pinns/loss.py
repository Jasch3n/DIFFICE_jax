import jax.numpy as jnp
import jax

# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


#%% loss for inferring isotropic viscosity

def loss_iso_create(predf, eqn_all, scale, lw, basal=False):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''
    is_basal = basal
    # print("loss_iso_create thinks is_basal is", is_basal)
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all
    # print("DEBUG: loss_iso_create thinks basal is", basal)

    # loss function used for the PINN training
    def loss_fun(params, data):
        # print("loss_fun thinks is_basal is", is_basal)
        # print("DEBUG: loss_fun thinks basal is", basal)
        # print("DEBUG: params length:", len(params))
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the velocity data and their position
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]

        # load the thickness data and their position
        xh_smp = data['smp'][2]
        h_smp = data['smp'][3]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # whole-domain coordinates for the grounded friction constraint
        x_pred = data['ocean_mask'][0]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        if is_basal: 
            c_pred = net(x_pred)[:,4:5]
            ocean_mask_whole = data['ocean_mask'][1]
        # print("DEBUG: u_pred shape:", jnp.shape(u_pred))
        # print("DEBUG: h_pred shape:", jnp.shape(h_pred))


        # calculate the residue of equation
        if is_basal:
            ocean_mask_col = data['col'][1]
        else:
            ocean_mask = None

        if is_basal:
            f_pred, f_pred_grounded, term = gov_eqn(net, x_col, scale, basal=is_basal)
            # e1term1 = term[:,0:1]
            # e2term1 = term[:,1:2]
            # e12term2 = term[:,2:3]
            # e1term3 = term[:,3:4]
            # e2term3 = term[:,4:5]
            # e1term4 = term[:,6:7]
            # e2term4 = term[:,7:8]
            # e1term3_grounded = term[:,8:9]
            # e2term3_grounded = term[:,9:10]
            # jax.debug.print("--------------------------")
            # jax.debug.print("")
            # jax.debug.print("viscous terms | xx={:.6f}, yy={:.6f}, xy={:.6f}",
            #                 ms_error(e1term1)[0], ms_error(e2term1)[0], ms_error(e12term2)[0])
            # jax.debug.print("basal traction terms | x={:.6f}, y={:.6f}",
            #                 ms_error(e1term4)[0], ms_error(e2term4)[0])
            # jax.debug.print("grav drive terms | xdir={:.6f}, ydir={:.6f}",
            #                 ms_error(e1term3_grounded)[0], ms_error(e2term3_grounded)[0])
            # jax.debug.print("eq residue | xdir={:.6f}, ydir={:.6f}",
            #                 ms_error(f_pred_grounded[:,0:1])[0], ms_error(f_pred_grounded[:,0:1])[0])
        else:
            f_pred, term = gov_eqn(net, x_col, scale, basal=is_basal)
        # print("DEBUG: f_pred shape:", jnp.shape(f_pred))
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err))
        # calculate the mean squared root error of equation
        if is_basal:
            eqn_err = ms_error(ocean_mask_col*f_pred + (1-ocean_mask_col)*f_pred_grounded)
        else:
            eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)
        # calculate friction coef for floating ice (constrain basal friction to grounded ice)
        if is_basal:
            grounded_err = ms_error(ocean_mask_whole * c_pred)
            # grounded_err = 0

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])
        if is_basal:
            grounded_weight = jnp.array([1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)
        if is_basal:
            loss_grounded = jnp.sum(grounded_err * grounded_weight)
            # loss_grounded = 0.

        # load the loss_ref
        loss_ref = loss_fun.lref
        # calculate the total loss
        # # group the loss of all conditions and equations
        if is_basal:
            loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + lw[2]*loss_grounded) / loss_ref
            loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_grounded]),
                                data_err, eqn_err, bd_err, grounded_err])
        else:
            loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd) / loss_ref
            loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                    data_err, eqn_err, bd_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    return loss_fun


#%% loss for inferring anisotropic viscosity

def loss_aniso_create(predf, eqn_all, scale, lw):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the data of normalization condition
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]
        xh_smp = data['smp'][2]
        h_smp = data['smp'][3]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:2]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]
        h_pred = net(xh_smp)[:, 2:3]

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err))
        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2)

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)
        loss_sp = jnp.sum(sp_err)

        # load the loss_ref
        loss_ref = loss_fun.lref
        # load the weight for the regularization loss
        wsp = loss_fun.wsp
        # define the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + wsp * loss_sp) / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_sp]),
                                data_err, eqn_err, bd_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    loss_fun.wsp = lw[2]
    return loss_fun