
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose


from jaxrk.rkhs import FiniteVec, inner, SiEdSpVec
from jaxrk.rkhs.sp_vector import cumsum_after_idxkern
from jaxrk.kern import GaussianKernel 

rng = np.random.RandomState(1)



kernel_setups = [
    GaussianKernel()
] 


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_FiniteVec(D = 1, kernel = kernel_setups[0], N = 10):
    X = rng.randn(N, D)
    rv = FiniteVec(kernel, X, np.ones(len(X)).astype(np.float32))
    rv2 = FiniteVec.construct_RKHS_Elem(kernel, rng.randn(N + 1, D), np.ones(N + 1).astype(np.float32))
    assert np.allclose(inner(rv, rv), rv.k(rv.inspace_points, rv.inspace_points)*np.outer(rv.prefactors, rv.prefactors)) , "Simple vector computation not accurate"
    assert np.allclose(inner(rv, rv2), (rv.k(rv.inspace_points, rv2.inspace_points)*np.outer(rv.prefactors, rv2.prefactors)).sum(1, keepdims=True)), "Simple vector computation not accurate"

    N = 4
    X = rng.randn(N, D)

    rv = FiniteVec(kernel, X, np.ones(len(X))/2, points_per_split = 2)
    el = FiniteVec.construct_RKHS_Elem(kernel, X, prefactors=np.ones(N))
    gram = el.k(el.inspace_points)
    assert np.allclose(inner(el, el), np.sum(gram))
    assert np.allclose(np.squeeze(inner(el, rv)), np.sum(gram, 1).reshape(-1,2).mean(1))


    rv = FiniteVec(kernel, X, np.ones(len(X))/2, points_per_split = 2)
    assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
                                               [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Balanced vector computation not accurate"
    
    vec = FiniteVec(kernel, np.array([(0.,), (1.,), (0.,), (1.,)]), prefactors=np.array([0.5, 0.5, 1./3, 2./3]), points_per_split=2)
    m, v = vec.normalized().get_mean_var()
    assert np.allclose(m.flatten(), np.array([0.5, 2./3]))
    assert np.allclose(v.flatten(), kernel.var + np.array([0.5, 2./3]) - m.flatten()**2)
    #rv = FiniteVec(kernel, X, np.ones(len(X))/2, row_splits = [0,2,4])
    #assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
    #                                           [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Ragged vector computation not accurate"


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
def test_Mean_var(D = 1, kernel = kernel_setups[0]):
    N = 4

   
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,)]), prefactors=np.ones(2)/2)
    for pref in [el.prefactors, 2*el.prefactors]:
        el.prefactors = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 0.5)
        assert np.allclose(v, kernel.var + 0.5 - m**2)
    
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,)]), prefactors=np.array([1./3, 2./3]))
    for pref in [el.prefactors, 2*el.prefactors]:
        el.prefactors = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 2./3)
        assert np.allclose(v, kernel.var + 2./3 - m**2)
    
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,), (2., )]), prefactors=np.array([0.2, 0.5, 0.3]))
    for pref in [el.prefactors, 2*el.prefactors]:
        el.prefactors = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 1.1)
        assert np.allclose(v, kernel.var + 0.5 + 0.3*4 - m**2)
    
    vec = FiniteVec(kernel, np.array([(0.,), (1.,), (0.,), (1.,)]), prefactors=np.array([0.5, 0.5, 1./3, 2./3]), points_per_split=2)
    m, v = vec.normalized().get_mean_var()
    assert np.allclose(m.flatten(), np.array([0.5, 2./3]))
    assert np.allclose(v.flatten(), kernel.var + np.array([0.5, 2./3]) - m.flatten()**2)




@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('idx_kernel', kernel_setups)
def test_SiEdSpVec(kernel, idx_kernel):
    x = np.linspace(0,10,10)
    f = 0.5
    y1 = np.sin(x*f)
    y2 = np.cos(x*f)

    #obs_points = np.hstack([y1, y2]).reshape((-1, 1))
    #fvec = SiEdSpVec(kernel, obs_points, np.cumsum(np.array([y1.size, y2.size])), idx_kernel)

    #### NO INDEX KERNEL #### 
    # - i.e. just a kernel on trajectories that does not account for a reference point

    # one process observed
    fvec = SiEdSpVec(kernel, y1[:2, None], np.cumsum(np.array([2])), None, use_subtrajectories=False)
    gram = fvec.k_obs(fvec.inspace_points)
    assert len(fvec) == 1
    #when summing: assert np.allclose(fvec.inner(), gram.sum(keepdims=True))
    assert np.allclose(fvec.inner(), gram.mean(keepdims=True))

    ## now with subtrajectories
    fvec = SiEdSpVec(kernel, y1[:2, None], np.cumsum(np.array([2])), None, use_subtrajectories=True)
    assert len(fvec) == 2
    #when summing: assert np.allclose(fvec.inner(), np.cumsum(np.cumsum(gram, axis=0), axis=1))
    assert np.allclose(fvec.inner(), np.cumsum(np.cumsum(gram, axis=0) / np.arange(1, gram.shape[0] + 1)[:, None], axis=1) / np.arange(1, gram.shape[0] + 1)[None, :])

    # two processes observed
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=False)
    gram = fvec.k_obs(fvec.inspace_points)
    assert len(fvec) == 2
    # when summing:
    # assert np.allclose(fvec.inner(), np.array([(gram[:2, :2].sum(), gram[:2, 2].sum()),
    #                                            (gram[2, :2].sum(), gram[2, 2])]))
    assert np.allclose(fvec.inner(), np.array([(gram[:2, :2].mean(), gram[:2, 2].mean()),
                                                (gram[2, :2].mean(), gram[2, 2])]))

    ## now with subtrajectories
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=True)
    gram_selects = [(gram[:1, :1], gram[:1, :2], gram[:1, 2:]),
                    (gram[:2, :1], gram[:2, :2], gram[:2, 2:]),
                    (gram[2:, :1], gram[2:, :2], gram[2:, 2:]) ]
    assert len(fvec) == 3
    #when summing: assert np.allclose(fvec.inner(), np.array([[col.sum() for col in row] for row in gram_selects]))
    assert np.allclose(fvec.inner(), np.array([[col.mean() for col in row] for row in gram_selects]))
    
    #### INDEX KERNEL ####

    fact = idx_kernel(np.arange(-2, 0).reshape((-1, 1)), np.zeros((1, 1))).flatten()
    fact = np.outer(fact, fact)
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), idx_kernel, use_subtrajectories=True)
    assert len(fvec) == 3
    #when summing: assert np.allclose(fvec.inner(), np.array([[(col * fact[-col.shape[0]:, -col.shape[1]:]).sum() for col in row] for row in gram_selects]))
    assert np.allclose(fvec.inner(), np.array([[(col * fact[-col.shape[0]:, -col.shape[1]:]).mean() for col in row] for row in gram_selects]))


    

def test_cumsum_after_idxkern():
    gram = np.arange(4).reshape((2, 2))
    idx = np.ones((2, 1))
    assert np.allclose(cumsum_after_idxkern(gram, idx), np.cumsum(gram, 0))
    assert np.allclose(cumsum_after_idxkern(gram, np.array([[-2,-1]]).T), np.array([-gram[0], -2*gram[0] - gram[1]]))