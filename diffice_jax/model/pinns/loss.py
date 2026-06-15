import jax.numpy as jnp
import jax
import jax.debug as jdb

# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)

def ma_error(diff):
    return jnp.mean(jnp.abs(diff), axis=0)

def u_mag(u):
    return jnp.sqrt(jnp.sum(jnp.square(u), 1)) 


def kfac_component_residual(diff, weight):
    """Scale pointwise residuals so their squared sum equals weighted MSE."""

    return jnp.sqrt(weight / diff.size) * diff.reshape(-1)

#%% loss for inferring isotropic viscosity

def loss_iso_create(predf, eqn_all, scale, lw, basal=False):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''
    # is_basal = basal
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
        if basal:
            s_smp = data['smp'][4]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        if basal:
            u_bd = data['bd'][1]
            h_bd = data['bd'][2]
            mu_bd = data['bd'][3]
        else:
            nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        if basal: 
            s_pred = net(xh_smp)[:, 3:4]

        eqn_fn = lambda x: gov_eqn(net, x, scale, basal=basal)
        bd_fn = lambda x, nn: front_eqn(net, x, nn, scale)
        f_pred, term = jax.vmap(eqn_fn, in_axes=(0,))(x_col)
        f_pred_bd, _ = jax.vmap(bd_fn, in_axes=(0, 0))(x_bd, nn_bd)
        if basal:
            # Get network predictions 
            gl_pred = net(x_bd)
            u_bd_pred = jnp.hstack((gl_pred[:, 0:1], gl_pred[:, 1:2]))
            h_bd_pred = gl_pred[:, 2:3].flatten()
            # s_bd_pred = gl_pred[:, 3:4].flatten()
            mu_bd_pred = gl_pred[:, 4:5].flatten()

            # Calculate boundary mismatch
            u_bd_err = ms_error(u_bd_pred - u_bd)
            h_bd_err = ms_error(h_bd_pred.flatten() - h_bd.flatten())
            mu_bd_err = ms_error(jnp.log(mu_bd_pred.flatten()) - jnp.log(mu_bd.flatten()))
            # jdb.print('u_bd_err shape = {x}', x=u_bd_err.shape)
            # jdb.print('h_bd_err shape = {x}', x=h_bd_err.shape)
            # jdb.print('mu_bd_err shape = {x}', x=mu_bd_err.shape)
            bd_err = jnp.hstack((u_bd_err, h_bd_err, mu_bd_err))
        else:
            f_bd = f_pred_bd


        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        if basal:
            data_s_err = ms_error(s_pred - s_smp)
            eps = 1e-7
            data_log_u_err = ms_error( jnp.log( (u_mag(u_pred)+eps) / (u_mag(u_smp)+eps) ))

        if basal:
            data_err = jnp.hstack((data_u_err, data_h_err, data_s_err))
        else:
            data_err = jnp.hstack((data_u_err, data_h_err))

        # calculate the mean squared root error of equation
        if basal:
            # e1_err = ma_error(f_pred[:,0:1] - f_pred[:,1:2])
            # e2_err = ma_error(f_pred[:,2:3] - f_pred[:,3:4])
            # eqn_err = jnp.hstack([e1_err, e2_err])
            eqn_err = ms_error(f_pred)
            # bd_eqn_err = ms_error(f_pred_bd)
            # visc_1 = term[:, 9:10]
            # grav_basal_1 = term[:, 10:11]
            # visc_2 = term[:, 11:12]
            # grav_basal_2 = term[:, 12:13]
            # mag_err = jnp.hstack([ms_error(jnp.abs(visc_1) - jnp.abs(grav_basal_1)), ms_error(jnp.abs(visc_2)-jnp.abs(grav_basal_2))])
        else:
            eqn_err = ms_error(f_pred)
        # calculate the mean squared root error of boundary condition
    
        if not basal:
            bd_err = ms_error(f_bd)

        # set the weight for each condition and equation
        if basal:
            data_weight = jnp.array([1., 1., 0.6, 0.6]) # include a weight for surface elevation
        else:
            data_weight = jnp.array([1., 1., 0.6])

        eqn_weight = jnp.array([1., 1.])
        if basal:
            mag_weight = jnp.array([1., 1.])

        if basal:
            # bd_weight = jnp.array([1., 1., 1., 0.5, 0.5])
            # bd_weight = jnp.array([1., 1., 1.])
            bd_weight = jnp.array([0.3, 0.3, 0.3, 1.0])
        else:
            bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        # if basal:
        #     loss_bd_eqn = jnp.sum(bd_eqn_err * eqn_weight)
        # if basal:
        #     loss_mag = jnp.sum(mag_err * mag_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)

        # load the loss_ref
        loss_ref = loss_fun.lref
        # calculate the total loss
        # # group the loss of all conditions and equations
        # loss = (lw[0]*loss_data + lw[1]*loss_eqn + lw[2]*loss_mag + lw[3]*loss_bd) / loss_ref
        # if basal:
        loss = (lw[0]*loss_data + lw[1]*loss_eqn + lw[2]*loss_bd) / loss_ref
        #     loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
        #                             data_err, eqn_err, bd_err])
        # elif not basal: # assume domain is all floating
        #     loss = (lw[0]*loss_data + lw[1]*loss_eqn + lw[2]*loss_bd) / loss_ref
            # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                data_err, eqn_err, bd_err])
            
        return loss, loss_info

    def kfac_residual_terms(params, data):
        """Return KFAC residual groups for data, equation, and calving-front terms."""

        net = lambda z: predf(params, z)
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]
        xh_smp = data['smp'][2]
        h_smp = data['smp'][3]
        x_col = data['col'][0]
        x_bd = data['bd'][0]

        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])

        data_residuals = [
            kfac_component_residual(u_pred[:, 0:1] - u_smp[:, 0:1], lw[0] * data_weight[0]),
            kfac_component_residual(u_pred[:, 1:2] - u_smp[:, 1:2], lw[0] * data_weight[1]),
            kfac_component_residual(h_pred - h_smp, lw[0] * data_weight[2]),
        ]

        eqn_fn = lambda x: gov_eqn(net, x, scale, basal=basal)[0]
        f_pred = jax.vmap(eqn_fn, in_axes=(0,))(x_col)
        eqn_residuals = [
            kfac_component_residual(f_pred[:, 0:1], lw[1] * eqn_weight[0]),
            kfac_component_residual(f_pred[:, 1:2], lw[1] * eqn_weight[1]),
        ]

        if basal:
            u_bd = data['bd'][1]
            h_bd = data['bd'][2]
            mu_bd = data['bd'][3]
            gl_pred = net(x_bd)
            bd_weight = jnp.array([0.3, 0.3, 0.3, 1.0])
            bd_residuals = [
                kfac_component_residual(gl_pred[:, 0:1] - u_bd[:, 0:1], lw[2] * bd_weight[0]),
                kfac_component_residual(gl_pred[:, 1:2] - u_bd[:, 1:2], lw[2] * bd_weight[1]),
                kfac_component_residual(gl_pred[:, 2:3] - h_bd.reshape(-1, 1), lw[2] * bd_weight[2]),
                kfac_component_residual(jnp.log(gl_pred[:, 4:5]) - jnp.log(mu_bd.reshape(-1, 1)), lw[2] * bd_weight[3]),
            ]
        else:
            nn_bd = data['bd'][1]
            bd_fn = lambda x, nn: front_eqn(net, x, nn, scale)[0]
            f_bd = jax.vmap(bd_fn, in_axes=(0, 0))(x_bd, nn_bd)
            bd_weight = jnp.array([1., 1.])
            bd_residuals = [
                kfac_component_residual(f_bd[:, 0:1], lw[2] * bd_weight[0]),
                kfac_component_residual(f_bd[:, 1:2], lw[2] * bd_weight[1]),
            ]

        return {
            "data": jnp.concatenate(data_residuals).reshape(-1, 1),
            "eqn": jnp.concatenate(eqn_residuals).reshape(-1, 1),
            "ct": jnp.concatenate(bd_residuals).reshape(-1, 1),
        }

    def kfac_residuals(params, data, terms=None):
        """Weighted residual vector matching ``loss_fun`` term composition."""

        residual_terms = kfac_residual_terms(params, data)
        active_terms = ("data", "eqn", "ct") if terms is None else terms
        residuals = [residual_terms[name] for name in active_terms]
        return jnp.concatenate(residuals).reshape(-1, 1)

    def kfac_objective(params, data, terms=None):
        """Squared residual objective used by KFAC."""

        residuals = kfac_residuals(params, data, terms=terms)
        return jnp.sum(jnp.square(residuals)) / loss_fun.lref

    loss_fun.lref = 1.0
    loss_fun.kfac_residual_terms = kfac_residual_terms
    loss_fun.kfac_residuals = kfac_residuals
    loss_fun.kfac_objective = kfac_objective
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
