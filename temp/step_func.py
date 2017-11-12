import numpy as np
from math import sqrt
from tqdm import tqdm as tqdm
            
from utils import sub_m, sub_0, solver_sub_0

from core import DelayedKeyboardInterrupt

# build lib to import!
from imc_sparse_ops.imc_ops import shrink_row


def step_qaadmm(X, W, Y, H, R, C, eta, core_loss,
                sparse=True, n_iterations=50, linearize=False,
                method="l-bfgs", rtol=1e-5, atol=1e-8):

    C_lasso, C_group, C_ridge = C
    Obj = core_loss(X, W, Y, H, R)

    LL, WW, ZZ = np.zeros_like(W), W.copy(), W.copy()

    iteration, n_sq_dim = 0, sqrt(np.prod(W.shape))
    resid_p_hist, resid_d_hist = [], []
    while iteration < n_iterations:
        # get the tolerances
        tol_p = n_sq_dim * atol + rtol * max(
            np.linalg.norm(WW.reshape(-1), 2),
            np.linalg.norm(ZZ.reshape(-1), 2))

        tol_d = n_sq_dim * atol + rtol * \
            np.linalg.norm(ZZ.reshape(-1), 2)

        # no need to copy WW or ZZ, their updates are not inplace
        WW_old, ZZ_old, LL_old = WW, ZZ, LL.copy()

        # admm steps
        ZZ = sub_m(WW + LL, C_lasso, C_group, C_ridge, eta=eta)
        
        if method == "cg":
            WW = W + sub_0((ZZ - W) - LL, Obj, eta=eta,
                           tol=1e-8, linearize=linearize)

        else:
            WW = solver_sub_0(ZZ - LL, Obj, x0=WW_old,
                              eta=eta, tol=1e-8,
                              method=method)
        # end if

        LL += WW - ZZ

        # residuals: primal and dual feasibility. 
        resid_p = np.linalg.norm((WW - ZZ).reshape(-1), 2)
        resid_d = np.linalg.norm((ZZ_old - ZZ).reshape(-1), 2) / eta

        iteration += 1

        resid_p_hist.append(resid_p)
        resid_d_hist.append(resid_d)

        if resid_p <= tol_p and resid_d <= tol_d:
            break
        # end if

    return ZZ if sparse else WW


def step_decoupled(X, W, Y, H, R, C, eta, core_loss,
                   debug=False, rtol=1e-5, atol=1e-8):
    if not isinstance(eta, list):
        eta = W.shape[0] * [eta]

    assert W.shape[0] == len(eta)

    C_lasso, C_group, C_ridge = C

    # The gradient of the loss at the current "solution" W
    Gt = core_loss(X, W, Y, H, R).grad()

    # Get a descent direction
    W_new = np.empty_like(W)
    for l, eta_l in zip(range(W.shape[0]), eta):
        # compute the descent direction
        W_new[l] = shrink_row(W[l] - Gt[l] * eta_l,
                              C_lasso * eta_l,
                              C_group * eta_l,
                              C_ridge * eta_l)

    return W_new

def step_adhoc(X, W, Y, H, R, C, eta, core_loss,
               rtol=1e-5, atol=1e-8):
    Obj = core_loss(X, W, Y, H, R)
    delta = sub_0(np.zeros_like(W), Obj, eta=eta,
                  tol=1e-8, linearize=False)
    return sub_m(delta + W, *C, eta=eta)

# =================================================================================================

# =================================================================================================

def loss(X, W, Y, H, R, C, core_loss):
    Obj = core_loss(X, W, Y, H, R)
    v_val = Obj.value()

    # The regularizers
    reg_ridge = 0.5 * (np.linalg.norm(W.reshape(-1), 2)**2) \
                + 0.5 * (np.linalg.norm(H.reshape(-1), 2)**2)

    reg_group = np.linalg.norm(W, 2, axis=-1).sum() \
                + np.linalg.norm(H, 2, axis=-1).sum()

    reg_lasso = np.linalg.norm(W.reshape(-1), 1) \
                + np.linalg.norm(H.reshape(-1), 1)

    C_lasso, C_group, C_ridge = C
    regul = C_group * reg_group + C_lasso * reg_lasso \
            + C_ridge * reg_ridge

    # Get the prediction for the full matrix
    XW, YH = np.dot(X, W), np.dot(Y, H)
    R_hat = np.dot(XW, YH.T)

    return R_hat, v_val, regul


def imc_by_qa(X, W, Y, H, R, R_full, core_loss, step_fn=step_qaadmm,
              C=(0.0, 1.0, 0.0), eta=1e-5, n_iterations=500,
              continuation=None, verbose=False,
              rtol=1e-5, atol=1e-8):

    # array to save all losses, problem is computing.
    loss_arr = []
    
    iteration = continuation or 0
    format_ = """%(observed)0.3e + %(reg).3e, %(full)0.3e; Score %(score).2f%%; """ \
            + """W %(W_sp).0f%%, H %(H_sp).0f%%; W %(div_W).3e, H %(div_H).3e"""

    n_sq_dim_W, n_sq_dim_H = sqrt(np.prod(W.shape)), sqrt(np.prod(H.shape))
    with tqdm(initial=iteration, total=n_iterations,
              disable=not verbose) as pbar:
        while iteration < n_iterations:
            try:
                tol_W = n_sq_dim_W * atol + rtol * \
                    np.linalg.norm(W.reshape(-1), 2)

                tol_H = n_sq_dim_H * atol + rtol * \
                    np.linalg.norm(H.reshape(-1), 2)

                W_old, H_old = W.copy(), H.copy()
                with DelayedKeyboardInterrupt():
                    # Solve for H holding W fixed
                    H = step_fn(Y, H, X, W, R.T, C, eta, core_loss,
                                rtol=rtol, atol=atol)

                    # Solve for W given H
                    W = step_fn(X, W, Y, H, R, C, eta, core_loss,
                                rtol=rtol, atol=atol)

                # Get the prediction for the full matrix
                R_hat, v_val, _reg = loss(X, W, Y, H, R, C, core_loss)
                _score = core_loss.score(R_hat, R_full) * 100
                _full = core_loss.v_func(R_hat, R_full).sum()
                _W_sp = 100 * np.isclose(W, 0).mean()
                _H_sp = 100 * np.isclose(H, 0).mean()

                div_W = np.linalg.norm((W_old - W).reshape(-1), 2)
                div_H = np.linalg.norm((H_old - H).reshape(-1), 2)

                pbar.postfix = format_ % {
                    "observed": v_val,
                    "reg":      _reg,
                    "score":    _score,
                    "full":     _full,
                    "W_sp":     _W_sp,
                    "H_sp":     _H_sp,
                    "div_W":    div_W,
                    "div_H":    div_H,
                }
                pbar.update(1)
                
                loss_arr.append([v_val, _reg, _score, _full,
                                 _W_sp, _H_sp, div_W, div_H])
                # return time per iteration

                iteration += 1

                if div_W <= tol_W and div_H <= tol_H:
                    break
                # end if

            except KeyboardInterrupt:
                break

    loss_arr = np.swapaxes(loss_arr, 0, 1)
    exp_type = ['Observed Elements', 'Regularization', 'Score',
                'Full Matrix', 'Zero Values of W', 'Zero Values of H',
                'L2-Norm Variation W', 'L2-Norm Variation H']
    norm_type = ['L2-Loss', 'L2-Loss', 'Score', 'L2-Loss',
                 '%', '%', 'L2-Norm', 'L2-Norm']
    
    return W, H, iteration, loss_arr, exp_type, norm_type