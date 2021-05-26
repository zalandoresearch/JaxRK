import jax.numpy as np
from jax import grad
from ..rkhs import FiniteVec
from jax.random import randint, PRNGKey
from scipy.optimize import minimize
from time import time
from numpy.random import randint

import numpy as onp

__all__ = ["frank_wolfe_pos_proj"]
def __casted_output(function):
    return lambda x: onp.asarray(function(x), dtype=np.float64)

def frank_wolfe_pos_proj(element:FiniteVec, solution:FiniteVec = None, num_samples:np.int32 = 100):
    assert(len(element) == 1)
    #key = PRNGKey(np.int32(time()))
    if solution is None:
        solution = FiniteVec(element.k, element.insp_pts[:1, :], np.zeros(1), points_per_split=1)
    for k in range(num_samples):
        def cost(x):
            x = x.reshape((1, -1))
            return (solution(x) - element(x)).sum()
        g_cost = grad(cost)
        idx = randint(0, element.points_per_split - 1)
        cand = element.insp_pts[idx:idx+1, :]
        #print(cand)
        #print(cost(cand), grad(cost)(cand))
        res = minimize(__casted_output(cost), cand, jac = __casted_output(g_cost))
        solution.inspace_points = np.vstack([solution.insp_pts, res["x"]])
        gamma_k = 1./(k + 1)
        solution.prefactors = np.hstack([(1-gamma_k) * solution.prefactors, gamma_k])
        solution.points_per_split = solution.points_per_split + 1
    return solution

def frank_wolfe_fx(element:FiniteVec, num_samples:np.int32 = 100):
    assert(len(element) == 1)
    #key = PRNGKey(np.int32(time()))
    solution = FiniteVec(element.k, element.insp_pts[:1, :], np.zeros(1), points_per_split=1)
    for k in range(num_samples):
        def cost(x):
            x = x.reshape((1, -1))
            return (solution(x) - element(x)).sum()
        g_cost = grad(cost)
        idx = randint(0, element.points_per_split - 1)
        cand = element.insp_pts[idx:idx+1, :]
        #print(cand)
        #print(cost(cand), grad(cost)(cand))
        res = minimize(__casted_output(cost), cand, jac = __casted_output(g_cost))
        solution.inspace_points = np.vstack([solution.insp_pts, res["x"]])
        gamma_k = 1./(k + 1)
        solution.prefactors = np.hstack([(1-gamma_k) * solution.prefactors, gamma_k])
        solution.points_per_split = solution.points_per_split + 1
    return solution
