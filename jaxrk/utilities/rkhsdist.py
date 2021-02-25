import jax.numpy as np

__all__ = ["rkhs_cdist", "rkhs_cdist_ignore_const"]

def rkhs_cdist(G_ab:np.array, G_a:np.array=None, G_b:np.array=None, power:float = 2.):
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
    return rkhs_cdist_unchecked(G_ab, G_a, G_b, power)

def rkhs_cdist_ignore_const(G_ab:np.array, G_b:np.array, power:float = 2.):
    sqdist = np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.:
        return sqdist
    else:
        return np.power(sqdist, power / 2.)

def rkhs_cdist_unchecked(G_ab:np.array, G_a:np.array, G_b:np.array, power:float = 2.):
    sqdist = np.diagonal(G_a)[:, np.newaxis] + np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.:
        return sqdist
    else:
        return np.power(sqdist, power / 2.)