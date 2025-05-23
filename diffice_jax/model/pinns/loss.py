import jax.numpy as jnp


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

    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all
    # print("DEBUG: loss_iso_create thinks basal is", basal)

    # loss function used for the PINN training
    def loss_fun(params, data, basal=basal):
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

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        if basal: 
            c_pred = net(x_smp)[:,4:5]
            ocean_mask_smp = data['smp'][4]
        # print("DEBUG: u_pred shape:", jnp.shape(u_pred))
        # print("DEBUG: h_pred shape:", jnp.shape(h_pred))


        # calculate the residue of equation
        if basal:
            ocean_mask = data['ocean_mask'][0]
        else:
            ocean_mask = None

        if basal:
            f_pred, f_pred_grounded, term = gov_eqn(net, x_col, scale, basal=True)
            print(jnp.shape(f_pred_grounded))
            print(jnp.shape(ocean_mask))
        else:
            f_pred, term = gov_eqn(net, x_col, scale, basal=False, ocean_mask=None)
        # print("DEBUG: f_pred shape:", jnp.shape(f_pred))
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err))
        # calculate the mean squared root error of equation
        if basal:
            eqn_err = ms_error(ocean_mask*f_pred + (1-ocean_mask)*f_pred_grounded)
        else:
            eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)
        # calculate friction coef for floating ice (constrain basal friction to grounded ice)
        if basal:
            grounded_err = ms_error(ocean_mask_smp * c_pred)
            # grounded_err = 0

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])
        if basal:
            grounded_weight = jnp.array([1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)
        if basal:
            loss_grounded = jnp.sum(grounded_err * grounded_weight)

        # load the loss_ref
        loss_ref = loss_fun.lref
        # calculate the total loss
        # # group the loss of all conditions and equations
        if basal:
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