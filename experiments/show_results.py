import numpy as np
import time

from ipywidgets import interact, Dropdown, fixed
from IPython.display import display

from utils import load, save
from sgimc.utils import plot_loss, performance,\
                        make_imc_data, sparsify
    
from sgimc import IMCProblem

from sgimc.qa_objective import QAObjectiveL2Loss,\
                               QAObjectiveLogLoss


def update_value(param_value, param_name, exp_setup, exp_struc, path):

    PROBLEM, STEP, mask_scale, scale, noise, seed, C, eta = exp_setup
    n_samples, n_objects, n_rank, n_features, K = exp_struc

    if PROBLEM == 'classification':
        QAObjectiveLoss = QAObjectiveLogLoss
    else:
        QAObjectiveLoss = QAObjectiveL2Loss

    if param_name == 'n_samples':
        # iterating by n_samples
        X, W_ideal, Y, H_ideal, R_full = make_imc_data(
            param_value, n_features, n_objects, n_features, n_rank,
            scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
    elif param_name == 'n_objects':
        #iterating by n_objects
        X, W_ideal, Y, H_ideal, R_full = make_imc_data(
            n_samples, n_features, param_value, n_features, n_rank,
            scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
    elif param_name == 'n_rank':
        # iterating by n_rank
        X, W_ideal, Y, H_ideal, R_full = make_imc_data(
            n_samples, n_features, n_objects, n_features, param_value,
            scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
    elif param_name == 'n_features':
        # iterating by n_features
        X, W_ideal, Y, H_ideal, R_full = make_imc_data(
            n_samples, param_value, n_objects, param_value, n_rank,
            scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
    elif param_name == 'K':
        # iterating by K
        X, W_ideal, Y, H_ideal, R_full = make_imc_data(
            n_samples, n_features, n_objects, n_features, n_rank,
            scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
        K = param_value

    R, mask = sparsify(R_full, mask_scale, random_state=seed)
    WH_name = 'WH_{}'.format(param_value) + '_' + param_name
    f_name = path + WH_name + '.gz'
    W, H = load(f_name)

    problem = IMCProblem(QAObjectiveLoss, X, Y, R, n_threads=8)
    loss_arr, exp_type, norm_type = performance(problem, W, H, C, R_full)

    plot_loss(loss_arr, exp_type, norm_type, fig_size=4, max_cols=4, yscale="log")



def update_param(param_name, exp_setup, exp_struc, path):
    b_name = path + param_name + '/' + 'is_iterable.pic'
    b_iter = load(b_name)
    if b_iter:
        path_upd = path + param_name + '/'
        f_name = path_upd + 'param.pic'
        data = load(f_name)

        interact(update_value,
                 param_value = Dropdown(options=data, description='Parameter value:'),
                 param_name = fixed(param_name),
                 exp_setup = fixed(exp_setup),
                 exp_struc = fixed(exp_struc),
                 path = fixed(path_upd)
            )
    else:
        print('No experiments to show!')



def show_results(path):
        
    setup_name = path + 'setup.pic'
    struc_name = path + 'standard_structure.pic'
    
    exp_setup = load(setup_name)
    exp_struc = load(struc_name)
    iter_name = ['n_samples', 'n_objects', 'n_rank', 'n_features', 'K']

    interact(update_param,
             param_name = Dropdown(options=iter_name, description='Parameter name:'),
             exp_setup = fixed(exp_setup),
             exp_struc = fixed(exp_struc),
             path = fixed(path)
        )