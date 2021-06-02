import jax.numpy as np
import scipy as osp
from numpy.random import rand

__all__ = ["rkhs_gram_cdist", "rkhs_gram_cdist_ignore_const", "choose_representer", "choose_representer_from_gram", "gram_projection"]

def rkhs_gram_cdist(G_ab:np.array, G_a:np.array=None, G_b:np.array=None, power:float = 2.):
    assert len(G_ab.shape)==2
    if G_a is not None:
        assert len(G_a.shape) == 2 and G_a.shape[0] == G_a.shape[1] and G_ab.shape[0] == G_a.shape[0]
        assert G_ab.shape[0] % G_a.shape[1] == 0, "Shapes of Gram matrices do not broadcast"
    if G_b is not None:
        assert G_b.shape[0] == G_b.shape[1] and G_ab.shape[1] == G_b.shape[1]
        assert G_ab.shape[1] % G_b.shape[0]  == 0, "Shapes of Gram matrices do not broadcast"
    if G_a is None or G_b is None:
        assert G_a == None and G_b == None, 'Either none or both of G_repr, G_orig should be None'
        assert np.all(G_ab == G_ab.T)
        #representer 
        G_a = G_b = G_ab
    return rkhs_gram_cdist_unchecked(G_ab, G_a, G_b, power)

def rkhs_gram_cdist_ignore_const(G_ab:np.array, G_b:np.array, power:float = 2.):
    sqdist = np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.:
        return sqdist
    else:
        return np.power(sqdist, power / 2.)

def rkhs_gram_cdist_unchecked(G_ab:np.array, G_a:np.array, G_b:np.array, power:float = 2.):
    sqdist = np.diagonal(G_a)[:, np.newaxis] + np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.:
        return sqdist
    else:
        return np.power(sqdist, power / 2.)


def choose_representer(support_points, factors, kernel):
    return choose_representer_from_gram(kernel(support_points).astype(np.float64), factors)

def choose_representer_from_gram(G, factors):
    fG = np.dot(factors, G)
    rkhs_distances_sq = (np.dot(factors, fG).flatten() + np.diag(G) - 2 * fG).squeeze()
    rval = np.argmin(rkhs_distances_sq)
    assert rval < rkhs_distances_sq.size
    return rval

def __casted_output(function):
    return lambda x: onp.asarray(function(x), dtype=np.float64)

def gram_projection(G_orig_repr:np.array,  G_orig:np.array=None, G_repr:np.array=None, method:str = "representer"):
    if method == "representer":
        return np.argmin(rkhs_gram_cdist(G_orig_repr, G_repr, G_orig), 0)
    elif method == "pos_proj":
        assert G_repr is not None
        s = G_orig_repr.shape
        n_pref = np.prod(np.array(s))
        def cost(M):
            M = M.reshape(s)
            return np.trace(rkhs_gram_cdist_ignore_const(G_orig_repr @ M.T, M @ G_repr@ M.T))

        res = osp.optimize.minimize(__casted_output(cost),
                               rand(n_pref)+ 0.0001,
                               jac = __casted_output(grad(cost)),
                               bounds = [(0., None)] * n_pref)
        return res["x"].reshape(s)
    else:
        assert False, "No valid method selected"