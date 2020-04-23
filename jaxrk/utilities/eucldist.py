import jax.numpy as np
from jax import jit

#based on https://stackoverflow.com/questions/52030458/vectorized-spatial-distance-in-python-using-numpy

__all__ = ["eucldist"]
@jit
def sqeucldist_simple(a, b = None):
    a_sumrows = np.einsum('ij,ij->i', a, a)
    if b is not None:        
        b_sumrows = np.einsum('ij,ij->i', b, b)
    else:
        b = a
        b_sumrows = a_sumrows
    return a_sumrows[:, np.newaxis] + b_sumrows - 2 * a @ b.T

@jit
def sqeucldist_extension(a, b = None):
    A_sq = a**2

    if b is not None:
        B_sq = b**2
    else:
        b = a
        B_sq = A_sq

    nA, dim = a.shape
    nB = b.shape[0]

    A_ext = np.hstack([np.ones((nA, dim)), a, A_sq])    
    B_ext = np.vstack([B_sq.T, -2.0*b.T, np.ones((dim, nB))])
    return A_ext @ B_ext

def eucldist(a, b = None, power = 1., variant = "simple"):
    if variant == "simple":
        sqdist = sqeucldist_simple(a, b)
    elif variant == "extension":
        sqdist = sqeucldist_extension(a, b)
    else:
        assert()

    if power == 2:
        return sqdist
    else:
        return np.power(np.clip(sqdist, 0.), power / 2.)
        

