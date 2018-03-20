import numpy as np
from tqdm import tqdm
import time

from sgimc.utils import make_imc_data, sparsify
from utils import load, save

from sgimc import IMCProblem, imc_descent

from sgimc.qa_objective import QAObjectiveL2Loss,\
                               QAObjectiveLogLoss
    
from sgimc.algorithm import admm_step
from sgimc.algorithm.decoupled import step as decoupled_step


def step_qaadmm(problem, W, H, C, eta, method="l-bfgs", sparse=True,
                n_iterations=50, rtol=1e-5, atol=1e-8):

    approx_type = "quadratic" if method in ("cg", "tron") else "linear"
    Obj = problem.objective(W, H, approx_type=approx_type)

    return admm_step(Obj, W, C, eta, sparse=sparse, method=method,
                     n_iterations=n_iterations, rtol=rtol, atol=atol)

def step_decoupled(problem, W, H, C, eta, rtol=1e-5, atol=1e-8):

    Obj = problem.objective(W, H, approx_type="linear")

    return decoupled_step(Obj, W, C, eta, rtol=rtol, atol=atol)

    
def build_experiment(PROBLEM, STEP, standard_structure,
                     samples, objects, ranks, features, Ks,
                     mask_scale, scale, noise, seed, path):
    
    # stand_str = (, , , , )
    # assertions
    assert PROBLEM in ('classification', 'regression')
    assert STEP in ('qaadmm', 'decoupled')


    # setting up global parameters for the experiment
    if PROBLEM == 'classification':
        QAObjectiveLoss = QAObjectiveLogLoss
        path_problem = 'class'
        C = 1e0, 1e-1, 1e-3
        eta = 1e0
    else:
        QAObjectiveLoss = QAObjectiveL2Loss
        path_problem = 'regres'
        C = 2e-3, 2e-4, 1e-4
        eta = 1e1
    
    if STEP == 'qaadmm':
        step_fn = step_qaadmm
        path_step = 'qaadmm'
    else: 
        step_fn = step_decoupled
        path_step = 'decoup'
        eta = 1e-3

    random_state = np.random.RandomState(seed)
    step_kwargs = {"C": C, "eta": eta, "rtol": 1e-5, "atol": 1e-8}


    # unpacking standard structure and hardcoding parameters for iterarions
    n_samples, n_objects, n_rank, n_features, K = standard_structure

    iter_name = ['n_samples', 'n_objects', 'n_rank', 'n_features', 'K']
    iter_data = (samples, objects, ranks, features, Ks)


    # saving global parameters (structure hardcoded for a while)   
    path_time = time.strftime('%Y%m%d_%H%M%S')
    PATH = path + 'experiment_' + path_problem + '_' + path_step + '_' + path_time + '/'

    setup_to_save = (PROBLEM, STEP, mask_scale,
                     scale, noise, seed, C, eta)
    save(setup_to_save, PATH, 'setup')
    save(standard_structure, PATH, 'standard_structure')


    # EXPERIMENTS
    for name, data, i in zip(iter_name, iter_data, np.arange(len(iter_data))):
        path_to_save = PATH + name + '/'
        if not data is None:
            save(True, path_to_save, 'is_iterable')
            save(data, path_to_save, 'param')
            for param_to_change in data:
                if i == 0:
                    # iterating by n_samples
                    X, W_ideal, Y, H_ideal, R_full = make_imc_data(
                        param_to_change, n_features, n_objects, n_features, n_rank,
                        scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
                elif i == 1:
                    #iterating by n_objects
                    X, W_ideal, Y, H_ideal, R_full = make_imc_data(
                        n_samples, n_features, param_to_change, n_features, n_rank,
                        scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
                elif i == 2:
                    # iterating by n_rank
                    X, W_ideal, Y, H_ideal, R_full = make_imc_data(
                        n_samples, n_features, n_objects, n_features, param_to_change,
                        scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
                elif i == 3:
                    # iterating by n_features
                    X, W_ideal, Y, H_ideal, R_full = make_imc_data(
                        n_samples, param_to_change, n_objects, param_to_change, n_rank,
                        scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
                elif i == 4:
                    # iterating by K
                    X, W_ideal, Y, H_ideal, R_full = make_imc_data(
                        n_samples, n_features, n_objects, n_features, n_rank,
                        scale=scale, noise=noise, binarize=(PROBLEM == 'classification'), random_state=seed)
                    K = param_to_change

                R, mask = sparsify(R_full, mask_scale, random_state=seed)
                        
                problem = IMCProblem(QAObjectiveLoss, X, Y, R, n_threads=8)

                W_0 = random_state.normal(size=(X.shape[1], K))
                H_0 = random_state.normal(size=(Y.shape[1], K))

                W, H = W_0.copy(), H_0.copy()

                W, H = imc_descent(
                    problem, W, H, step_fn, step_kwargs=step_kwargs,
                    n_iterations=1000, return_history=True,
                    rtol=1e-4, atol=1e-7, verbose=True)

                WH_name = 'WH_{}'.format(param_to_change) + '_' + name

                save((W, H), path_to_save, WH_name, gz=9)
        else:
            save(False, path_to_save, 'is_iterable')