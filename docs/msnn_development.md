# MSNN Development 

Yesterday, we implemented a basic structure for the MSNN in diffice. Today, I want to make modifications. Firstly, let us change the way we estimate the error magnitude epsilon. There is in fact no need to estimate the error using frequency because we have actual data to compare with. The deviation of the X-PINN solution from the true solution can be estimated by the data loss. Here is an example of how one can do it. 

```
import jax.numpy as jnp
import numpy as np
from jax.extend.backend import get_backend

from jax.tree_util import tree_map
from scipy.io import loadmat

from scipy.interpolate import griddata

from diffice_jax import normdata_xpinn, dsample_xpinn
from diffice_jax import vectgrad, ssa_iso, dbc_iso
from diffice_jax import init_xpinn, solu_xpinn
from diffice_jax import loss_iso_xpinn
from diffice_jax import predict_xpinn
from diffice_jax import adam_opt, lbfgs_opt

DATA_MAT = os.path.join('/Users/jiapchen/Research/PinningPointInversion/Data/StokesComparison/ProcessedXPinnData', MODEL_FOLDER)
rawdata=loadmat(DATA_MAT)

# Unpack everything we need
trained_params = d['trained_params']
data_all       = d['data_all']
basal_mask     = d['basal_mask']

# ── Reconstruct forward prediction ──
print("  Reconstructing forward prediction ...")
pred_u, _ = solu_xpinn(scale, basal_mask=basal_mask)
f_u = lambda x, idx: pred_u(trained_params, x, idx)
func_all = [f_u, ssa_iso]
results = predict_xpinn(func_all, data_all, posi_all, idxcrop_all,
                        idxgall, basal_mask=basal_mask)

scale = tree_map(lambda x: data_all[x][4][0:2], [0, 1])
xd_1 = rawdata['xd'][0, 1]
yd_1 = rawdata['yd'][0, 1]
xd_2 = rawdata['xd'][0, 2]
yd_2 = rawdata['yd'][0, 2]

def norm_coords(x, y):
    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2
    y_mean = jnp.mean(y)
    y_range = (y.max() - y.min()) / 2
    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range

    return jnp.hstack((x_n, y_n))

def load_data_to_dict(rawdata, idx):
    res = {}
    res['u'] = rawdata['ud'][0][idx].flatten()
    res['v'] = rawdata['vd'][0][idx].flatten()
    res['h'] = rawdata['hd'][0][idx].flatten()
    return res 

def normalize_data_to_dict(rawdata, scale, idx):
    dmean, drange = scale[idx-1][0:2]
    print(dmean)
    xm, ym, um, vm, hm = dmean[0:5]
    lx0, ly0, u0, v0 = drange[0:4]

    ud = rawdata['ud'][0][idx].flatten()
    vd = rawdata['vd'][0][idx].flatten()
    hd = rawdata['hd'][0][idx].flatten()

    res = {} 
    res['u'] = np.array((ud-um)/u0).flatten() 
    res['v'] = np.array((vd-vm)/v0).flatten() 
    res['h'] = np.array(hd/hm).flatten() 
    # if len(dmean > 5):
    #     res['s'] = np.array(U[:,3] * dmean[5]).flatten()

    res['u0']=u0
    res['v0']=v0
    res['h0']=hm 
    res['um']=um 
    res['vm']=vm
    
    return res

def load_pred_to_dict(U):
    res = {} 
    res['u'] = np.array(U[:,0]).flatten() 
    res['v'] = np.array(U[:,1]).flatten() 
    res['h'] = np.array(U[:,2]).flatten()
    
    return res

def denormalize(U, scale):
    dmean, drange = scale[0:2]
    print(dmean)
    xm, ym, um, vm, hm = dmean[0:5]
    lx0, ly0, u0, v0 = drange[0:4]
    res = {} 
    res['u'] = np.array(U[:,0]*u0 + um).flatten() 
    res['v'] = np.array(U[:,1]*v0 + vm).flatten() 
    res['h'] = np.array(U[:,2]*hm).flatten() 
    if len(dmean > 5):
        res['s'] = np.array(U[:,3] * dmean[5]).flatten()

    res['u0']=u0
    res['v0']=v0
    res['h0']=hm 
    res['um']=um 
    res['vm']=vm
    
    return res

X1 = norm_coords(xd_1, yd_1)
X2 = norm_coords(xd_2, yd_2)

# The pred u has basal_mask baked into it, no need to specify basal
net_1 = lambda x: pred_u(trained_params, x, 0)
net_2 = lambda x: pred_u(trained_params, x, 1)
U_1   = net_1(X1)
U_2   = net_2(X2)
res1 = load_pred_to_dict(U_1)
res2 = load_pred_to_dict(U_2)
res1_d = normalize_data_to_dict(rawdata, scale, 1)
res2_d = normalize_data_to_dict(rawdata, scale, 2)

# The SSA equation needs to know whether the region is grounded or not 
f_1, terms_1 = ssa_iso(net_1, X1, scale[0])
f_2, terms_2 = ssa_iso(net_2, X2, scale[1], basal=True)

eps_u1 = np.sqrt(np.mean(np.square(res1['u'] - res1_d['u'])))
eps_v1 = np.sqrt(np.mean(np.square(res1['v'] - res1_d['v'])))
eps_h1 = np.sqrt(np.mean(np.square(res1['h'] - res1_d['h'])))

eps_u2 = np.sqrt(np.mean(np.square(res2['u'] - res2_d['u'])))
eps_v2 = np.sqrt(np.mean(np.square(res2['v'] - res2_d['v'])))
eps_h2 = np.sqrt(np.mean(np.square(res2['h'] - res2_d['h'])))
```
Then, instead of normalizing the data for the correction stages by one single epsilon, each variable can be normalized independently using its own epsilon value. At inference, then, each of these variables will be also scaled by their respective epsilons. 