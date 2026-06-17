import pickle
from ..model.pinns.networks import solu_create as solu_pinn
from ..model.xpinns.networks import solu_create as solu_xpinn
from ..model.architecture import resolve_architecture

def save_model(filepath, params, model_type="xpinn", **config):
    """
    Saves the trained weights and the network architecture configuration.
    
    Args:
        filepath: Path to save the .pkl file
        params: Trained parameters
        model_type: "pinn" or "xpinn"
        **config: Architecture settings (e.g. embedding, architecture, use_modified_mlp, scl, act_s, basal_mask, scale)
    """
    try:
        import jax
        params = jax.device_get(params)
        config = jax.device_get(config)
    except Exception:
        pass

    model_dict = {
        'params': params,
        'model_type': model_type,
        'config': config
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)

#[TODO]: Change the embedding input here, as it is a temporary fix 
def load_model(filepath, return_elements=False, embedding=False):
    """
    Loads the trained model configuration and parameters, and returns an encapsulated predictive function.
    
    Args:
        filepath: Path to the .pkl file
        return_elements: If True, also returns the raw (params, config) unpacked from the dictionary so metadata can be accessed.
    
    Returns:
       predict_fn(x, idx) for xpinns or predict_fn(x) for pinns.
       If return_elements=True, returns `(predict_fn, params, config)`
    """
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
        
    params = model_dict['params']
    model_type = model_dict['model_type']
    config = model_dict['config']

    # Load flags from config if they exist, otherwise use default/passed arguments
    cfg = {
        'scl': config.get('scl', 1),
        'act_s': config.get('act_s', 0),
        'embedding': config.get('embedding', embedding),
        'use_rwf': config.get('use_rwf', False), 
        'use_modified_mlp': config.get('use_modified_mlp', False),
        'architecture': resolve_architecture(
            architecture=config.get('architecture'),
            use_modified_mlp=config.get('use_modified_mlp', False),
        ),
    }

    if model_type == "xpinn":
        # Required for xpinns logic
        cfg['basal_mask'] = config.get('basal_mask', None)
        scale = config.get('scale', None)
        if scale is None:
            raise ValueError("scale must be provided in config for xpinns")
            
        solNN = solu_xpinn(scale, **cfg)
        f_pred = solNN[0]
        grad_fn = solNN[1]
        
        def predict_fn(x, idx):
            return f_pred(params, x, idx)  # Returns uvh, mu, c
            
        if return_elements:
            # We return solNN too so user has access to gradf
            return predict_fn, params, config, solNN
        return predict_fn

    elif model_type == "pinn":
        cfg['basal'] = config.get('basal', False)
        # pinn doesn't require scale parameter in solu_pinn
        solNN = solu_pinn(**cfg)
        def predict_fn(x):
            return solNN(params, x) 
            
        if return_elements:
            return predict_fn, params, config
        return predict_fn
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
