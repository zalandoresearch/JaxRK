
import jax.numpy as np
from  jax.numpy import log, exp
import numpy.random as npr
from  jax.numpy.linalg import inv, cholesky
from  jax.scipy.special import multigammaln, gammaln, logsumexp
from  jax.scipy import stats as stats
from  scipy.stats import uniform

__all__ = ["location_mixture_logpdf", "mixt"]

def location_mixture_logpdf(samps, locations, location_weights, distr_at_origin, contr_var = False, variant = 1):
    diff = samps - locations[:, np.newaxis, :]
    lpdfs = distr_at_origin.logpdf(diff.reshape([np.prod(diff.shape[:2]), diff.shape[-1]])).reshape(diff.shape[:2])
    logprop_weights = log(location_weights/location_weights.sum())[:, np.newaxis]
    return logsumexp(lpdfs + logprop_weights, 0)

class categorical(object):
    def __init__(self, p):
        self.lp = log(np.array(p)).flatten()
        if np.abs(1-exp(logsumexp(self.lp))) >= 10**-7:
            raise ValueError("the probability vector does not sum to 1")
        self.cum_lp = np.array([logsumexp(self.lp[:i]) for i in range(1, len(self.lp)+1)])
    
    def get_num_unif(self):
        return 1    
        
    def ppf(self, x, indic = False, x_in_logspace = False):
        if not x_in_logspace:
            x = log(x)
        idc = (self.cum_lp >= x)
        if indic:
            idc[np.argmax(idc)+1:] = False
            return idc
        else:
            return np.argmax(idc)
    
    def logpdf(self, x, indic = False):
        if indic:
            x = np.argmax(x, 1)
        return self.lp[x]
            
    def rvs(self, size = 1, indic = False):
        assert(size >= 0)
        return np.array([self.ppf(uniform.rvs(), indic = indic)
                             for _ in range(size)])

class mixt(object):
    def __init__(self, dim, comp_dists, comp_weights):
        self.dim = dim
        self.comp_dist = comp_dists
        self.dist_cat = categorical(comp_weights)

    
    def logpdf(self, x):
        comp_logpdf = np.array([self.dist_cat.logpdf(i)+ self.comp_dist[i].logpdf(x)
                              for i in range(len(self.comp_dist))])
        rval = logsumexp(comp_logpdf, 0)
        if len(comp_logpdf.shape) > 1:
            rval = rval.reshape((rval.size, 1))
        return rval
    
    def rvs(self, num_samples=1):
        rval = np.array([self.comp_dist[i].rvs(1) for i in self.dist_cat.rvs(num_samples)])
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]
        else:
            return rval