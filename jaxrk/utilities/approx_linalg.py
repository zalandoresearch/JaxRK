import jax.numpy as np
import numpy.random as random

def nystrom_eigh(gram, n_comp, regul = 0.) -> tuple:
    assert len(gram.shape) == 2
    assert gram.shape[0] == gram.shape[1]
    assert gram.shape[0] >= n_comp

    perm = np.arange(gram.shape[0])
    idx_in = perm[:n_comp]
    idx_out = perm[n_comp:]
    λ, vec_in = np.linalg.eigh(gram[idx_in, :][:,idx_in])
    vec_out = gram[idx_out, :][:, idx_in] @ vec_in @ np.diag(1./(λ+regul))
    return (vec_in, vec_out, λ)

def nystrom_inv(gram, n_comp, regul = 0.) -> np.array:
    p = random.permutation(gram.shape[0])
    ip = np.argsort(p)
    (vec_in, vec_out, λ) = nystrom_eigh(gram[p,:][:,p], n_comp, regul)
    vec = np.vstack([vec_in, vec_out])
    rval =  vec@ np.diag(1./(λ+regul))@vec.T
    return rval[ip,:][:,ip]