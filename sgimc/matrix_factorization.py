import numpy as np
from numba import njit
from math import sqrt


@njit
def mf_sgd_step(user_idx, item_idx, feedback, P, Q, lrate, reg):
    '''SGD optimization function.'''
    
    cum_error = 0
    for k in range(len(feedback)):
        a = feedback[k]
        i = user_idx[k]
        j = item_idx[k]

        pi = P[i, :]
        qj = Q[j, :]

        e = a - np.dot(pi, qj)

        new_pi = pi + lrate * (e*qj - reg*pi)
        new_qj = qj + lrate * (e*pi - reg*qj)

        P[i, :] = new_pi
        Q[j, :] = new_qj

        cum_error += e*e
    return cum_error


def unbiased_matrix_factorization(R_train,
                                  rank=10, lrate=0.005, reg=0.05, num_epochs=100, tol=1e-4,
                                  seed=None, verbose=False):
    '''
    Function to compute weighted matrix factorization (MF) based on a
    train dataset `R_train`. Given be the model: `R_hat = P.dot(Q.T)`.
    
    ---------------------------------------
    Parameters:
    
    R_train: (must be sparse matrix) matrix of training dataset,
    rank: rank of factors,
    lrate: SGD step,
    reg: regularization coefficient,
    num_epoch: number of iteration of SGD,
    tol: tolerance of stopping criteria
    
    ---------------------------------------
    Returns:
    Two matricies (factors): P, Q.
    '''
    
    # TODO: add assertions
    # TODO: patient tolerance
    # TODO: learning rate schedule
    
    # converting to the COO representation
    R_train = R_train.tocoo()
    interactions = [R_train.row, R_train.col, R_train.data]
    n_users, n_items = R_train.shape
    
    # factors initialization
    random_state = np.random.RandomState(seed) if seed else np.random
    P = random_state.normal(scale=0.1, size=(n_users, rank))
    Q = random_state.normal(scale=0.1, size=(n_items, rank))
        
    last_err = np.finfo(np.float64).max
    for epoch in range(num_epochs):
        # SGD step
        new_err = mf_sgd_step(*interactions, P, Q, lrate, reg)
        err_delta = abs(last_err - new_err) / last_err
        
        if verbose:
            # error of the current step
            rmse = sqrt(new_err / len(interactions[2]))
            print('Epoch {} RMSE: {}'.format(epoch+1, rmse))
        
        last_err = new_err
        if err_delta < tol: break
            
    return P, Q
