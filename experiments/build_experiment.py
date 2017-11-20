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

    approx_type = "quadratic" if method in ("cg",) else "linear"
    Obj = problem.objective(W, H, approx_type=approx_type)

    return admm_step(Obj, W, C, eta, sparse=sparse, method=method,
                     n_iterations=n_iterations, rtol=rtol, atol=atol)

def step_decoupled(problem, W, H, C, eta, rtol=1e-5, atol=1e-8):

    Obj = problem.objective(W, H, approx_type="linear")

    return decoupled_step(Obj, W, C, eta, rtol=rtol, atol=atol)

    
def build_experiment(PROBLEM, STEP, samples, objects,
                     ranks, features, Ks, mask_scale,
                     scale, noise, seed, path):
    
    
    # assertions
    assert PROBLEM in ('classification', 'regression')
    assert STEP in ('qaadmm', 'decoupled')
    
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
        
    path_time = time.strftime('%Y%m%d_%H%M%S')
        
    PATH = path + 'experiment_' + path_problem\
           + '_' + path_step + '_' + path_time + '/'

    params_to_save = (PROBLEM, STEP, samples, objects,
                     ranks, features, Ks, mask_scale,
                     scale, noise, seed, C, eta)
    save(params_to_save, PATH, 'params')
    
    random_state = np.random.RandomState(seed)
    step_kwargs = {"C": C, "eta": eta, "rtol": 1e-5, "atol": 1e-8}
    
    for n_samples in samples:
        for n_objects in objects:
            for n_rank in ranks:
                for n_features in features:
                    for K in Ks:
                        X, W_ideal, Y, H_ideal, R_full \
                        = make_imc_data(
                            n_samples, n_features, n_objects, n_features,
                            n_rank, scale=scale, noise=noise,
                            binarize=(PROBLEM == 'classification'),
                            random_state=seed)

                        R, mask = sparsify(R_full, mask_scale, random_state=seed)
                        
                        problem = IMCProblem(QAObjectiveLoss, X, Y, R, n_threads=8)

                        W_0 = random_state.normal(size=(X.shape[1], K))
                        H_0 = random_state.normal(size=(Y.shape[1], K))

                        W, H = W_0.copy(), H_0.copy()

                        step_fn = step_qaadmm

                        W, H = imc_descent(
                            problem, W, H, step_fn, step_kwargs=step_kwargs,
                            n_iterations=1000, return_history=True,
                            rtol=1e-4, atol=1e-7, verbose=True)
                        
                        WH_name = 'WH_{}x{}-{}rk-{}ft-{}K'.format(
                            n_samples, n_objects, n_rank, n_features, K)
                        
                        save((W, H), PATH, WH_name, gz=9)
    
    #end for for for for for
    
    