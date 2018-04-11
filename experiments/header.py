import numpy as np
from tqdm import tqdm

from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sgimc.utils import sparsify, sparsify_with_mask, make_imc_data

from sgimc import IMCProblem

from sgimc.qa_objective import QAObjectiveL2Loss
from sgimc.qa_objective import QAObjectiveLogLoss
from sgimc.qa_objective import QAObjectiveHuberLoss

from sgimc.algorithm.admm import sub_0_cg
from sgimc.algorithm.admm import sub_0_lbfgs
from sgimc.algorithm.admm import sub_m

from sgimc import imc_descent

from utils import calculate_loss, invert,   \
                  accuracy, get_prediction, \
                  rmse, relative_loss

random_state = np.random.RandomState(0x0BADCAFE)


from sgimc.algorithm import admm_step
from sgimc.algorithm.decoupled import step as decoupled_step

def step_qaadmm(problem, W, H, C, eta, method="l-bfgs", sparse=True,
                n_iterations=50, rtol=1e-5, atol=1e-8):

    approx_type = "quadratic" if method in ("cg","tron",) else "linear"
    Obj = problem.objective(W, H, approx_type=approx_type)

    return admm_step(Obj, W, C, eta, sparse=sparse, method=method,
                     n_iterations=n_iterations, rtol=rtol, atol=atol)

def step_decoupled(problem, W, H, C, eta, rtol=1e-5, atol=1e-8):

    Obj = problem.objective(W, H, approx_type="linear")

    return decoupled_step(Obj, W, C, eta, rtol=rtol, atol=atol)

import warnings
warnings.filterwarnings('ignore')