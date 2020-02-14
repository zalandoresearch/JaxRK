# Copyright 2018 the kernelflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

import probkern
from probkern.rkhs import FiniteVec, inner, Elem, SiEdSpVec
from probkern.rkhs.sp_vector import cumsum_after_idxkern
from probkern.kern import GaussianKernel 

rng = np.random.RandomState(1)



kernel_setups = [
    GaussianKernel()
] 


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_FiniteVec(D, kernel, N):
    X = rng.randn(N, D)
    rv = FiniteVec(kernel, X, np.ones(len(X)).astype(np.float))
    rv2 = Elem(kernel, rng.randn(N + 1, D), np.ones(N + 1).astype(np.float))
    assert np.allclose(inner(rv, rv), rv.k(rv.inspace_points, rv.inspace_points)*np.outer(rv.prefactors, rv.prefactors)) , "Simple vector computation not accurate"
    assert np.allclose(inner(rv, rv2), (rv.k(rv.inspace_points, rv2.inspace_points)*np.outer(rv.prefactors, rv2.prefactors)).sum(1, keepdims=True)), "Simple vector computation not accurate"

    N = 4
    X = rng.randn(N, D)

    rv = FiniteVec(kernel, X, np.ones(len(X))/2, points_per_split = 2)
    assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
                                               [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Balanced vector computation not accurate"
    
    #rv = FiniteVec(kernel, X, np.ones(len(X))/2, row_splits = [0,2,4])
    #assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
    #                                           [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Ragged vector computation not accurate"


@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_FiniteVec(D, kernel, N):
    X = rng.randn(N, D)
    rv = FiniteVec(kernel, X, np.ones(len(X)).astype(np.float))
    rv2 = Elem(kernel, rng.randn(N + 1, D), np.ones(N + 1).astype(np.float))
    assert np.allclose(inner(rv, rv), rv.k(rv.inspace_points, rv.inspace_points)*np.outer(rv.prefactors, rv.prefactors)) , "Simple vector computation not accurate"
    assert np.allclose(inner(rv, rv2), (rv.k(rv.inspace_points, rv2.inspace_points)*np.outer(rv.prefactors, rv2.prefactors)).sum(1, keepdims=True)), "Simple vector computation not accurate"

    N = 4
    X = rng.randn(N, D)

    rv = FiniteVec(kernel, X, np.ones(len(X))/2, points_per_split = 2)
    assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],
                                               [np.mean(rv.k(X[:2,:], X[2:,:])), np.mean(rv.k(X[2:,:]))]])), "Balanced vector computation not accurate"
    
    #rv = FiniteVec(kernel, X, np.ones(len(X))/2, row_splits = [0,2,4])
    #assert np.allclose(inner(rv, rv), np.array([[np.mean(rv.k(X[:2,:])), np.mean(rv.k(X[:2,:], X[2:,:]))],

@pytest.mark.parametrize('D', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
def test_Elem(D, kernel ):
    N = 4
    X = rng.randn(N, D)
    rv = FiniteVec(kernel, X, np.ones(len(X))/2, points_per_split = 2)
    el = FiniteVec.construct_RKHS_Elem(kernel, X, prefactors=np.ones(N))
    gram = el.k(el.inspace_points)
    assert np.allclose(inner(el, el), np.sum(gram))
    assert np.allclose(np.squeeze(inner(el, rv)), np.sum(gram, 1).reshape(-1,2).mean(1))

    el = FiniteVec.construct_RKHS_Elem(kernel, np.array([(0.,), (1.,)]), prefactors=np.ones(2)/2)
    m, v = el.get_mean_var()
    print(m,v, kernel.var + 0.5 - m**2)
    assert np.allclose(m, 0.5)
    assert np.allclose(v, kernel.var + 0.5 - m**2)



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
    assert np.allclose(fvec.inner(), gram.sum(keepdims=True))

    ## now with subtrajectories
    fvec = SiEdSpVec(kernel, y1[:2, None], np.cumsum(np.array([2])), None, use_subtrajectories=True)
    assert np.allclose(fvec.inner(), np.cumsum(np.cumsum(gram, axis=0), axis=1))

    # two processes observed
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=False)
    gram = fvec.k_obs(fvec.inspace_points)
    assert np.allclose(fvec.inner(), np.array([(gram[:2, :2].sum(), gram[:2, 2].sum()),
                                               (gram[2, :2].sum(), gram[2, 2])]))

    ## now with subtrajectories
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), None, use_subtrajectories=True)
    gram_selects = [(gram[:1, :1], gram[:1, :2], gram[:1, 2:]),
                    (gram[:2, :1], gram[:2, :2], gram[:2, 2:]),
                    (gram[2:, :1], gram[2:, :2], gram[2:, 2:]) ]
    assert np.allclose(fvec.inner(), np.array([[col.sum() for col in row] for row in gram_selects]))
    
    #### INDEX KERNEL ####

    fact = idx_kernel(np.arange(-2, 0).reshape((-1, 1)), np.zeros((1, 1))).flatten()
    fact = np.outer(fact, fact)
    fvec = SiEdSpVec(kernel, y1[:3, None], np.cumsum(np.array([2, 1])), idx_kernel, use_subtrajectories=True)
    assert np.allclose(fvec.inner(), np.array([[(col * fact[-col.shape[0]:, -col.shape[1]:]).sum() for col in row] for row in gram_selects]))


    

def test_cumsum_after_idxkern():
    gram = np.arange(4).reshape((2, 2))
    idx = np.ones((2, 1))
    assert np.allclose(cumsum_after_idxkern(gram, idx), np.cumsum(gram, 0))
    assert np.allclose(cumsum_after_idxkern(gram, np.array([[-2,-1]]).T), np.array([-gram[0], -2*gram[0] - gram[1]]))