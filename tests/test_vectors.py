
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose


from jaxrk.rkhs import FiniteVec, inner, SpVec
from jaxrk.kern import GaussianKernel 
from jaxrk.reduce import Prefactors, BalancedSum, LinearReduce

rng = np.random.RandomState(1)



kernel_setups = [
    GaussianKernel()
] 


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_FiniteVec(D, kernel, N):
    X = rng.randn(N, D)
    rv = FiniteVec(kernel, X, [Prefactors(np.ones(len(X)).astype(np.float32))])
    rv2 = FiniteVec.construct_RKHS_Elem(kernel, rng.randn(N + 1, D), np.ones(N + 1).astype(np.float32))
    assert np.allclose(inner(rv, rv), rv.k(rv.inspace_points, rv.inspace_points)*np.outer(rv.reduce[0].prefactors, rv.reduce[0].prefactors)) , "Simple vector computation not accurate"
    assert np.allclose(inner(rv, rv2), (rv.k(rv.inspace_points, rv2.inspace_points)*np.outer(rv.reduce[0].prefactors, rv2.reduce[0].linear_map)).sum(1, keepdims=True)), "Simple vector computation not accurate"

    N = 4
    X = rng.randn(N, D)

    rv = FiniteVec(kernel, X, [Prefactors(np.ones(len(X))/2), BalancedSum(2)])
    el = FiniteVec.construct_RKHS_Elem(kernel, X, np.ones(N))
    gram = el.k(el.inspace_points)
    assert np.allclose(inner(el, el), np.sum(gram))
    assert np.allclose(np.squeeze(inner(el, rv)), np.sum(gram, 1).reshape(-1,2).mean(1))


    rv = FiniteVec(kernel, X, [Prefactors(np.ones(len(X))/2), BalancedSum(2)])
    assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
                                               [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Balanced vector computation not accurate"
    
    vec = FiniteVec(kernel, np.array([(0.,), (1.,), (0.,), (1.,)]), [LinearReduce(np.array([0.5, 0.5, 0,0,0,0, 1./3, 2./3]).reshape((2, -1)))])
    m, v = vec.normalized().get_mean_var()
    assert np.allclose(m.flatten(), np.array([0.5, 2./3]))
    assert np.allclose(v.flatten(), kernel.var + np.array([0.5, 2./3]) - m.flatten()**2)
    #rv = FiniteVec(kernel, X, np.ones(len(X))/2, row_splits = [0,2,4])
    #assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
    #                                           [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Ragged vector computation not accurate"


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
def test_Mean_var(D, kernel):
    N = 4

   
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,)]), prefactors=np.ones(2)/2)
    for pref in [el.reduce[0].linear_map, 2*el.reduce[0].linear_map]:
        el.reduce[0].linear_map = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 0.5)
        assert np.allclose(v, kernel.var + 0.5 - m**2)
    
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,)]), prefactors=np.array([1./3, 2./3]))
    for pref in [el.reduce[0].linear_map, 2*el.reduce[0].linear_map]:
        el.reduce[0].linear_map = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 2./3)
        assert np.allclose(v, kernel.var + 2./3 - m**2)
    
    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,), (2., )]), prefactors=np.array([0.2, 0.5, 0.3]))
    for pref in [el.reduce[0].linear_map, 2*el.reduce[0].linear_map]:
        el.reduce[0].linear_map = pref
        m, v = el.normalized().get_mean_var()
        #print(m,v)
        assert np.allclose(m, 1.1)
        assert np.allclose(v, kernel.var + 0.5 + 0.3*4 - m**2)
    
    vec = FiniteVec(kernel, np.array([(0.,), (1.,), (0.,), (1.,)]), [LinearReduce(np.array([0.5, 0.5, 0, 0, 0, 0, 1./3, 2./3]).reshape((2, -1)))])
    m, v = vec.normalized().get_mean_var()
    assert np.allclose(m.flatten(), np.array([0.5, 2./3]))
    assert np.allclose(v.flatten(), kernel.var + np.array([0.5, 2./3]) - m.flatten()**2)
