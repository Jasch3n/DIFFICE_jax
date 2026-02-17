import jax.numpy as jnp
import jax

# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)

def ma_error(diff):
    return jnp.mean(jnp.abs(diff), axis=0)

def u_mag(u):
    return jnp.sqrt(jnp.sum(jnp.square(u), 1)) 

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
            mu_bd = data['bd'][2]
        else:
            nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        if basal: 
            s_pred = net(xh_smp)[:, 3:4]

        f_pred, term = gov_eqn(net, x_col, scale, basal=basal)
        f_pred_bd, _ = gov_eqn(net, x_bd, scale, basal=basal)
        if basal:
            gl_pred = net(x_bd)
            # mu_bd_pred = net(x_bd)[:,4:5].flatten()
            u_bd_pred = jnp.hstack((gl_pred[:, 0:1], gl_pred[:, 1:2]))
            mu_bd_pred = gl_pred[:, 4:5].flatten()
            mu_bd_err = ms_error(jnp.log(mu_bd_pred) - jnp.log(mu_bd))
            u_bd_err = ms_error(u_bd_pred - u_bd)
            # eqn_bd_err = ms_error(f_pred_bd)
            bd_err = mu_bd_err
        else:
            f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)


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
            bd_weight = jnp.array([1.])
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
        if basal:
            loss = (lw[0]*loss_data + lw[1]*loss_eqn + lw[2]*loss_bd) / loss_ref
            loss_info = [jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, data_log_u_err, 0]),
                                    data_err, eqn_err, bd_err, 0]), f_pred]
        elif not basal: # assume domain is all floating
            loss = (lw[0]*loss_data + lw[1]*loss_eqn + lw[2]*loss_bd) / loss_ref
            # group the loss of all conditions and equations
            loss_info = [jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                    data_err, eqn_err, bd_err]), f_pred]
            
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