#!/usr/bin/env python
import pickle
import jax.numpy as jnp
from dmff.eann.eann import eannprm_trans_f2i, eannprm_trans_i2f

def eannprm_trans_i2f(params):
    params_f = {}
    list_keys = ['w', 'b']
    n_layers = len(params['w'])
    for k in params.keys():
        if k in list_keys:
            for i_layer in range(n_layers):
                params_f[k + '.%d'%i_layer] = params[k][i_layer]
        else:
            params_f[k] = params[k]
    return params_f

def eannprm_trans_f2i(params):
    params_i = {}
    list_keys = ['w', 'b']
    for k in list_keys:
        params_i[k] = []
    # make sure keys are in order
    keys = list(params.keys())
    keys.sort()
    for k in keys:
        new_k = k.split('.')[0]
        if new_k in list_keys:
            params_i[new_k].append(params[k])
        else:
            params_i[k] = params[k]
    return params_i


if __name__ == '__main__':
    ifn = 'params-92.pickle'
    with open(ifn, 'rb') as ifile:
        params = pickle.load(ifile)
    
    params_i = params['energy']
    
    params_f = eannprm_trans_i2f(params_i)
    
    map_kname = {
            'rs': 'density.rs',
            'inta': 'density.inta',
            'c': 'density.params',
            'initpot': 'nnmod.initpot'
            }
    for k in map_kname:
        k_new = map_kname[k]
        params_f[k_new] = params_f[k]
        params_f.pop(k, None)

    keys = ['w.0', 'w.3', 'w.6']
    for k in params_f.keys():
        if k in keys:
             params_f[k] = params_f[k].swapaxes(1, 2)
    
    with open('params.pickle', 'wb') as ofile:
        pickle.dump(params_f, ofile)

