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
    

def update_exp(n_samples, n_objects, n_rank, n_features, K,
               PROBLEM, STEP, mask_scale, scale,
               noise, seed, C, eta, PATH):
    
    if PROBLEM == 'classification':
        QAObjectiveLoss = QAObjectiveLogLoss
    else:
        QAObjectiveLoss = QAObjectiveL2Loss
        
    X, W_ideal, Y, H_ideal, R_full \
        = make_imc_data(
            n_samples, n_features, n_objects, n_features,
            n_rank, scale=scale, noise=noise,
            binarize=(PROBLEM == "classification"),
            random_state=seed)

    R, mask = sparsify(R_full, mask_scale, random_state=seed)
    
    WH_name = 'WH_{}x{}-{}rk-{}ft-{}K'.format(
              n_samples, n_objects, n_rank, n_features, K)
    _filename = PATH + WH_name + '.gz'
    W, H = load(_filename)
    
    problem = IMCProblem(QAObjectiveLoss, X, Y, R, n_threads=8)
    
    loss_arr, exp_type, norm_type \
        = performance(problem, W, H, C, R_full)
        
    plot_loss(loss_arr, exp_type, norm_type,
              fig_size=4, max_cols=4, yscale="log")


def show_results(path):
        
    f_name = path + 'params.pic'
    
    PROBLEM, STEP, samples, objects,\
    ranks, features, Ks, mask_scale,\
    scale, noise, seed, C, eta        = load(f_name)
    
    interact(update_exp,
             n_samples    = Dropdown(options=samples, description='Number of samples:'),
             n_objects    = Dropdown(options=objects, description='Number of objects:'),
             n_rank       = Dropdown(options=ranks, description='Rank of matrix:'),
             n_features   = Dropdown(options=features, description='Number of features:'),
             K            = Dropdown(options=Ks, description='Assumed rank (K):'),
             PROBLEM     = fixed(PROBLEM),
             STEP        = fixed(STEP),
             mask_scale  = fixed(mask_scale),
             scale       = fixed(scale),
             noise       = fixed(noise),
             seed        = fixed(seed),
             C           = fixed(C),
             eta         = fixed(eta),
             PATH = fixed(path)
             )