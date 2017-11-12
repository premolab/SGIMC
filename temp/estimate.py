import numpy as np
from math import sqrt

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from step_func import step_qaadmm, step_decoupled,\
                      loss, imc_by_qa 

from core import QAObjective,\
                 QAObjectiveL2Loss,\
                 QAObjectiveLogLoss,\
                 QAObjectiveHuberLoss



def estimate(step, problem,
             n_samples=199, n_objects=201, n_rank=5, n_features=15,
             seed=0x0BADCAFE, noise=True, noise_scale=1e-3, K=10,
             C=(2e-3, 2e-4, 0), eta=1e0, mask_scale = 0.1,
             n_iterations=500):
    
    random_state = np.random.RandomState(seed)
    
    # creating 
    U = random_state.normal(scale=1/20., size=(n_samples, n_rank))
    V = random_state.normal(scale=1/20., size=(n_objects, n_rank))

    X = np.concatenate([
        U, random_state.normal(scale=1/20., size=(n_samples, n_features - n_rank))], axis=-1)
    Y = np.concatenate([
        V, random_state.normal(scale=1/20., size=(n_objects, n_features - n_rank))], axis=-1)

    R_full = np.dot(U, V.T)

    epsilon = random_state.normal(scale=noise_scale, size=R_full.shape)
    if noise:
        R_full += epsilon

    W_ideal = np.concatenate([np.eye(n_rank), np.zeros((n_features - n_rank, n_rank))], axis=0)
    H_ideal = np.concatenate([np.eye(n_rank), np.zeros((n_features - n_rank, n_rank))], axis=0)
    
# =========== whats the purpose ? ============================
    PX, s, QX = np.linalg.svd(X, full_matrices=1)
    PY, s, QY = np.linalg.svd(Y, full_matrices=1)

    proj_X = np.dot(PX[:, :n_features], PX[:, :n_features].T)
    proj_Y = np.dot(PY[:, :n_features], PY[:, :n_features].T)

    if not noise:
        assert np.allclose(np.dot(proj_X, R_full), R_full)
        assert np.allclose(np.dot(proj_Y, R_full.T), R_full.T)
# ============================================================
    
    if problem == "classification":
        QAObjectiveLoss = QAObjectiveLogLoss
        loss_name = 'LogLoss'
        R_full = (2. * (R_full > 0)) - 1
    elif problem == "regression":
        QAObjectiveLoss = QAObjectiveL2Loss
        loss_name = 'L2-Loss'
    else:
        assert problem in ("classification", "regression")
        
    if step == "qaadmm":
        step_fn = step_qaadmm
    elif step == "decoupled":
        step_fn = step_decoupled
    else:
        assert step in ("qaadmm", "decoupled")
        
    mask = random_state.uniform(size=R_full.shape) < mask_scale

    R_coo = coo_matrix((R_full[mask], np.nonzero(mask)),
                       shape=R_full.shape, dtype=np.float)
    R = R_coo.tocsr()
    
    # initialization
    W_0 = random_state.normal(size=(X.shape[1], K))
    H_0 = random_state.normal(size=(Y.shape[1], K))
    
    # run
    W, H = W_0.copy(), H_0.copy()
    iteration = None
    
    W, H, iteration, loss_arr, exp_type, norm_type = \
        imc_by_qa(
                          X, W.copy(), Y, H.copy(), R, R_full,
                          core_loss=QAObjectiveLoss, step_fn=step_fn,
                          C=C, eta=eta, n_iterations=n_iterations,
                          continuation=iteration, verbose=True,
                          rtol=1e-5, atol=1e-8
                 )
    return loss_arr, exp_type, norm_type
    