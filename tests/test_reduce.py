
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose


from jaxrk.rkhs import FiniteVec, inner, SpVec
from jaxrk.reduce import SparseReduce, SparseBlockReduce, LinearReduce

rng = np.random.RandomState(1)




def test_SparseReduce():
    gram = rng.randn(4, 3)
    r = SparseReduce([ np.array([0, 1]),
                       np.array([0, 3]),
                       np.array([0, 2]) ],
                     True) 
    rgr = r(gram, 0)
    assert np.allclose(rgr[0], (gram[0] + gram[1]) / 2) 
    assert np.allclose(rgr[1], (gram[0] + gram[3]) / 2) 
    assert np.allclose(rgr[2], (gram[0] + gram[2])/2)

def test_reduce_from_unique():
  inp = np.array([1,1,0,3,5,0])
  un1, cts1, red1 = SparseReduce.sum_from_unique(inp)
  un2, cts2, red2 = SparseBlockReduce.sum_from_unique(inp)
  un3, cts3, red3 = LinearReduce.sum_from_unique(inp)
  
  args = np.argsort(un2)

  i_out = np.outer(inp, inp)
  assert np.all(red1.reduce_first_ax(i_out) == red2.reduce_first_ax(i_out)[args, :])
  assert np.all(red1.reduce_first_ax(i_out) == red3.reduce_first_ax(i_out))

def test_LinearReduce():
    gram = rng.randn(4, 3)
    r = LinearReduce(np.array([(1, 1, 1, 1),
                               (0.5, 0.5, 2, 2)]))
    rgr = r(gram, 0)
    assert np.allclose(rgr[0], gram.sum(0)) 
    assert np.allclose(rgr[1], gram[:2].sum(0)/2 + gram[2:].sum(0)*2) 