
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose


from jaxrk.rkhs import FiniteVec, inner, SpVec
from jaxrk.kern import GaussianKernel 

rng = np.random.RandomState(1)



kernel_setups = [
    GaussianKernel()
] 


@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('idx_kernel', kernel_setups)
def test_SpVec(kernel, idx_kernel):
    assert("adapt this test from SiEdSpVec to SpVec (in particular: add index dimensions!)")
    x = np.linspace(0,10,10)
    f = 0.5
    y1 = np.sin(x*f)
    y2 = np.cos(x*f)

    #obs_points = np.hstack([y1, y2]).reshape((-1, 1))
    #fvec = SiEdSpVec(kernel, obs_points, np.cumsum(np.array([y1.size, y2.size])), idx_kernel)

    #### NO INDEX KERNEL #### 
    # - i.e. just a kernel on trajectories that does not account for a reference point

    # one process observed
    fvec = SpVec(kernel, y1[:2, None], np.cumsum(np.array([2])), None, use_subtrajectories=False)
    gram = fvec.k(fvec.inspace_points)
    assert len(fvec) == 1
    #when summing: assert np.allclose(fvec.inner(), gram.sum(keepdims=True))
    assert np.allclose(fvec.inner(), gram.mean(keepdims=True))

    ## now with subtrajectories
    fvec = SpVec(kernel, y1[:2, None], np.cumsum(np.array([2])), None, use_subtrajectories=True)
    assert len(fvec) == 2
    #when summing: assert np.allclose(fvec.inner(), np.cumsum(np.cumsum(gram, axis=0), axis=1))
    assert np.allclose(fvec.inner(), np.cumsum(np.cumsum(gram, axis=0) / np.arange(1, gram.shape[0] + 1)[:, None], axis=1) / np.arange(1, gram.shape[0] + 1)[None, :])

    # two processes observed
    fvec = SpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=False)
    gram = fvec.k_obs(fvec.inspace_points)
    assert len(fvec) == 2
    # when summing:
    # assert np.allclose(fvec.inner(), np.array([(gram[:2, :2].sum(), gram[:2, 2].sum()),
    #                                            (gram[2, :2].sum(), gram[2, 2])]))
    assert np.allclose(fvec.inner(), np.array([(gram[:2, :2].mean(), gram[:2, 2].mean()),
                                                (gram[2, :2].mean(), gram[2, 2])]))

    ## now with subtrajectories
    fvec = SpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=True)
    gram_selects = [(gram[:1, :1], gram[:1, :2], gram[:1, 2:]),
                    (gram[:2, :1], gram[:2, :2], gram[:2, 2:]),
                    (gram[2:, :1], gram[2:, :2], gram[2:, 2:]) ]
    assert len(fvec) == 3
    #when summing: assert np.allclose(fvec.inner(), np.array([[col.sum() for col in row] for row in gram_selects]))
    assert np.allclose(fvec.inner(), np.array([[col.mean() for col in row] for row in gram_selects]))
    
    #### INDEX KERNEL ####

    fact = idx_kernel(np.arange(-2, 0).reshape((-1, 1)), np.zeros((1, 1))).flatten()
    fact = np.outer(fact, fact)
    fvec = SpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), idx_kernel, use_subtrajectories=True)
    assert len(fvec) == 3
    #when summing: assert np.allclose(fvec.inner(), np.array([[(col * fact[-col.shape[0]:, -col.shape[1]:]).sum() for col in row] for row in gram_selects]))
    assert np.allclose(fvec.inner(), np.array([[(col * fact[-col.shape[0]:, -col.shape[1]:]).mean() for col in row] for row in gram_selects]))