from copy import copy
from jaxrk.reduce.centop_reductions import CenterInpFeat, DecenterOutFeat
from jaxrk.reduce.lincomb import LinearReduce
from jaxrk.reduce.base import Prefactors, Sum
from typing import Generic, TypeVar, Callable

import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from jaxrk.rkhs.vector import FiniteVec, inner
from jaxrk.core.typing import AnyOrInitFn

from .base import LinOp, RkhsObject, Vec

InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

#The following is input to a LinOp RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec) 

CombT = TypeVar("CombT", "FiniteOp[RhInpVectT, InpVecT]", InpVecT, np.array)


class FiniteOp(LinOp[InpVecT, OutVecT]):
    """Finite rank LinOp in RKHS
    """
    inp_feat:InpVecT
    outp_feat:OutVecT
    matr_init:AnyOrInitFn
    normalize:bool = False

    def setup(self, ):
        self.matr = self.const_or_param("matr", self.matr_init)

    def __len__(self):
        return len(self.inp_feat) * len(self.outp_feat)

    def __matmul__(self, right_inp:CombT) -> RkhsObject:
        if isinstance(right_inp, FiniteOp):
            matr = self.matr @ self.inp_feat.inner(right_inp.outp_feat) @ right_inp.matr
            rval = FiniteOp(right_inp.inp_feat, self.outp_feat, matr)
            return rval
        else:
            if isinstance(right_inp, DeviceArray):
                right_inp = FiniteVec(self.inp_feat.k, np.atleast_2d(right_inp))  
            lin_LinOp = (self.matr @ inner(self.inp_feat, right_inp)).T
            if self._normalize:
                lin_LinOp = lin_LinOp / lin_LinOp.sum(1, keepdims = True)
            r = [LinearReduce(lin_LinOp)]
            if len(right_inp) == 1:
                r.append(Sum())
            rval = self.outp_feat.extend_reduce(r)
            return rval
    
    def inner(self, Y:"FiniteOp[InpVecT, OutVecT]"=None, full=True):
        assert NotImplementedError("This implementation as to be tested")
        if Y is None:
            Y = self
        G_i = self.inp_feat.inner(Y.inp_feat)
        G_o = self.outp_feat.inner(Y.outp_feat)

        # check the following expression again
        if Y.matr.size < self.matr.size:
            return np.sum((G_o.T @ self.matr @ G_i) * Y.matr)
        else:
            return np.sum((G_o @ Y.matr @ G_i.T) * self.matr)

        # is the kronecker product taken the right way around or do self and Y have to switch plaches?
        #return self.reduce_gram(Y.reduce_gram(G_i.T.reshape(1, -1) @ np.kron(self.matr, Y.matr) @ G_o.reshape(1, -1), 1), 0)
    
    
    def reduce_gram(self, gram, axis = 0):
        return gram
    
    @property
    def T(self) -> FiniteOp[OutVecT, InpVecT]:
        return FiniteOp(self.outp_feat, self.inp_feat, self.matr.T, self.normalize)

    def __call__(self, inp:DeviceArray) -> RkhsObject:
        return self @ FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
    
    def apply(self, inp:CombT) -> RkhsObject:
        return self @ inp
    
    def solve(self, result:OutVecT) -> RkhsObject:
        raise NotImplementedError()



def CrossCovOp(inp_feat:InpVecT, outp_feat:OutVecT) -> FiniteOp[InpVecT, OutVecT]:
    assert len(inp_feat) == len(outp_feat)
    return FiniteOp(inp_feat, outp_feat, np.eye(len(inp_feat)))

def CovOp(inp_feat:InpVecT) -> FiniteOp[InpVecT, InpVecT]:
    return FiniteOp(inp_feat, inp_feat, np.eye(len(inp_feat)))
    
def CovOp_from_Samples(kern, inspace_points, prefactors = None) -> FiniteOp[InpVecT, InpVecT]:
    return CovOp(FiniteVec(kern, inspace_points, prefactors))
    
def Cov_regul(nsamps:int, nrefsamps:int = None, a:float = 0.49999999999999, b:float = 0.49999999999999, c:float = 0.1):
        """Compute the regularizer based on the formula from the Kernel Conditional Density operators paper (Schuster et al., 2020, Corollary 3.4).
        
        smaller c => larger bias, tighter stochastic error bounds
        bigger  c =>  small bias, looser stochastic error bounds

        Args:
            nsamps (int): Number of samples used for computing the RKHS embedding.
            nrefsamps (int, optional): Number of samples used for computing the reference distribution covariance operator. Defaults to nsamps.
            a (float, optional): Parameter a. Assume a > 0 and a < 0.5, defaults to 0.49999999999999.
            b (float, optional): Parameter b. Assume a > 0 and a < 0.5, defaults to 0.49999999999999.
            c (float, optional): Bias/variance tradeoff parameter. Assume c > 0 and c < 1, defaults to 0.1.

        Returns:
            [type]: [description]
        """
        if nrefsamps is None:
            nrefsamps = nsamps
            
        assert(a > 0 and a < 0.5)
        assert(b > 0 and b < 0.5)
        assert(c > 0 and c < 1)
        assert(nsamps > 0 and nrefsamps > 0)
        
        return max(nrefsamps**(-b*c), nsamps**(-2*a*c))

def Cov_inv(cov:FiniteOp[InpVecT, InpVecT], regul:float = None) -> "FiniteOp[InpVecT, InpVecT]":
    """Compute the inverse of this covariance operator with a certain regularization.

    Args:
        regul (float, optional): Regularization parameter to be used. Defaults to self.regul.

    Returns:
        FiniteOp[InpVecT, InpVecT]: The inverse operator
    """
    assert regul is not None
    inv_gram = np.linalg.inv(inner(cov.inp_feat) + regul * np.eye(len(cov.inp_feat), dtype = cov.matr.dtype))
    matr = (cov.matr @ cov.matr @ inv_gram @ inv_gram)
    rval = CovOp(cov.inp_feat)
    rval.matr = matr
    rval._inv = cov
        
    return rval
    
def Cov_solve(cov:FiniteOp[InpVecT, InpVecT], lhs:CombT, regul:float = None) -> RkhsObject:
    """If `inp` is an RKHS vector of length 1 (a mean embedding): Solve the inverse problem to find dP/dρ from equation
    μ_P = C_ρ dP/dρ
    where C_ρ is the covariance operator passed as `cov`, ρ is the reference distribution, and μ_P is given by `lhs`.
    If `lhs` is a `FiniteOp`: Solve the inverse problem to find operator B from equation
    A = C_ρ B
    where C_ρ is the covariance operator passed as `cov`, and A is given by `lhs`.
    
    Args:
        lhs (CombT): The embedding of the distribution of interest, or the LinOp of interest.
    """
    
    if isinstance(lhs, FiniteOp):
        reg_inp = lhs.outp_feat
    else:
        if isinstance(lhs, DeviceArray):
            lhs = FiniteVec(cov.inp_feat.k, np.atleast_2d(lhs))
        reg_inp = lhs
    if regul is None:
        regul = Cov_regul(max(reg_inp.nsamps().min(), 1), max(cov.inp_feat.nsamps(True), 1))
    return (Cov_inv(cov, regul) @ lhs)

def Cmo(inp_feat:InpVecT, outp_feat:OutVecT, regul:float = 0.01) -> FiniteOp[InpVecT, OutVecT]:
        regul = np.array(regul, dtype=np.float32)

        #we do not center the output features - this still leads to the correct results in the output of the CME
        c_inp_feat = inp_feat
        G = inner(c_inp_feat)
        assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)       
        matr = np.linalg.inv(G + regul * np.eye(len(inp_feat)))
        return  FiniteOp(inp_feat, outp_feat, matr)

def Cdo(inp_feat:InpVecT, outp_feat:OutVecT, ref_feat:OutVecT, regul = 0.01,) -> FiniteOp[InpVecT, OutVecT]:
        mo = Cmo(inp_feat, outp_feat, regul)
        matr = Cov_solve(CovOp(ref_feat), mo, regul).matr
        return FiniteOp(mo.inp_feat,
                         ref_feat,
                         matr,
                         normalize=True)
