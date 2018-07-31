"""Various utility functions and classes to help with experimentation."""
import numpy as np

from sgimc.utils import sparsify_with_mask


def add_noise_features(X, n_features, scale, random_state, return_sparse=True):
    
    if n_features == 0:
        return X
    
    n = X.shape[0]
    X_noise = random_state.normal(scale=scale, size=(n, n_features))
    X_comb = np.concatenate((X.toarray(), X_noise), axis=1)
    
    if return_sparse:
        X_comb = sparsify_with_mask(X_comb, X_comb != 0)
        
    return X_comb
