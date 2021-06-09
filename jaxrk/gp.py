from .rkhs import FiniteVec
from .core.typing import Array
import jax.numpy as np, flax.linen as ln, jax.scipy as scipy
from typing import Any

class GP(object):
    def __init__(self, x:FiniteVec, y:Array, noise:float, amp:float, kconst:float = 0.):
        self.x, self.y, self.noise, self.amp, self.kconst = ( x, y, noise, amp, kconst)
        self.ymean = np.mean(y)
        self.y -= self.ymean
        train_cov = self.amp * (self.x.inner() + kconst) + np.eye(len(self.x)) * self.noise 
        self.chol = scipy.linalg.cholesky(train_cov, lower=True)
        self.kinvy = scipy.linalg.solve_triangular(self.chol.T, scipy.linalg.solve_triangular(self.chol, self.y, lower=True))
    
    def marginal_likelihood(self):
        log2pi = np.log(2. * np.pi)
        ml = np.sum(
            -0.5 * np.dot(self.y.T, self.kinvy) -
            np.sum(np.log(np.diag(self.chol))) -
            (len(self.x) / 2.) * log2pi)
        ml -= np.sum(-0.5 * log2pi - np.log(self.amp)**2) # lognormal prior
        return -ml

    def predict(self, xtest:FiniteVec = None):
        cross_cov = self.amp*(self.x.inner(xtest) + self.kconst)
        mu = np.dot(cross_cov.T, self.kinvy) + self.ymean
        v = scipy.linalg.solve_triangular(self.chol, cross_cov, lower=True)
        var = (self.amp * (xtest.inner() + self.kconst) - np.dot(v.T, v))
        return mu, var
    
    def post_pred_likelihood(self, xtest:FiniteVec, ytest:Array):
        m, v = self.predict(xtest)
        return m, v, scipy.stats.multivariate_normal.logpdf(ytest.ravel(), m.ravel(), v)