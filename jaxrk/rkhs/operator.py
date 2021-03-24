from copy import copy
from jaxrk.reduce.centop_reductions import CenterInpFeat, DecenterOutFeat
from jaxrk.reduce.lincomb import LinearReduce
from jaxrk.reduce.base import Prefactors, Sum
from typing import Generic, TypeVar, Callable

import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from jaxrk.rkhs.vector import FiniteVec, inner

from .base import LinOp, RkhsObject, Vec

InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

#The following is input to a LinOp RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec) 

CombT = TypeVar("CombT", "FiniteOp[RhInpVectT, InpVecT]", InpVecT, np.array)


class FiniteOp(LinOp[InpVecT, OutVecT]):
    """Finite rank LinOp in RKHS
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, matr:np.array, mean_center_inp:bool = False, decenter_outp:bool = False, normalize = False, outp_bias:np.array = None):
        assert not (decenter_outp and outp_bias is not None), "Either decenter_outp == True or outp_bias != None, but not both"
        self.matr = matr
        self.mean_center_inp = mean_center_inp

        if not mean_center_inp:
            self.inp_feat = inp_feat
            self.outp_feat = outp_feat
        else:
            self.inp_feat = inp_feat.extend_reduce([CenterInpFeat(inp_feat.inner())])
            self.outp_feat = outp_feat
        self._normalize = normalize

        self.debias_outp = decenter_outp
        if decenter_outp:            
            outp_bias = np.ones((1, len(outp_feat)) ) / len(outp_feat)
        else:
            outp_bias = np.zeros((1, len(outp_feat)))

        if outp_bias is not None:
            assert outp_bias.shape[1] == len(outp_feat)
            if len(outp_bias.squeeze().shape) == 1:
                self.bias = outp_bias.squeeze()[np.newaxis, :]
            else:
                assert outp_bias.shape[0]  == len(outp_feat) 
                self.bias = outp_bias
    
    def __len__(self):
        return len(self.inp_feat)

    def __matmul__(self, right_inp:CombT) -> RkhsObject:
        if isinstance(right_inp, FiniteOp):
            G = inner(self.inp_feat, right_inp.outp_feat)
            
            if not right_inp.debias_outp:
                matr = self.matr @ G @ right_inp.matr
                inp_bias = (matr @ right_inp.bias.T).T
            else:
                matr = self.matr @ (G - G @ right_inp.bias.T) @ right_inp.matr
                inp_bias = (self.matr @ G @ right_inp.bias.T).T

            rval = FiniteOp(right_inp.inp_feat, self.outp_feat, matr, outp_bias=self.bias + inp_bias)
            rval.mean_center_inp = right_inp.mean_center_inp
            return rval
        else:
            if isinstance(right_inp, DeviceArray):
                right_inp = FiniteVec(self.inp_feat.k, np.atleast_2d(right_inp))  
            lin_LinOp = (self.matr @ inner(self.inp_feat, right_inp)).T
            if self.debias_outp:
                r = [DecenterOutFeat(lin_LinOp)]
            else:
                if self._normalize:
                    lin_LinOp = lin_LinOp / lin_LinOp.sum(1, keepdims = True)
                r = [LinearReduce(lin_LinOp + self.bias)]
            if len(right_inp) == 1:
                r.append(Sum())
            rval = self.outp_feat.extend_reduce(r)
            return rval

    def __call__(self, inp:DeviceArray) -> RkhsObject:
        return self @ FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
    
    def apply(self, inp:CombT) -> RkhsObject:
        return self @ inp
    
    def solve(self, result:OutVecT) -> RkhsObject:
        raise NotImplementedError()



class CrossCovOp(FiniteOp[InpVecT, OutVecT]):
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul = 0.01):
        super().__init__(inp_feat, outp_feat, np.diag((inp_feat.prefactors + outp_feat.prefactors)/2))
        assert len(inp_feat) == len(outp_feat)
        assert np.allclose(inp_feat.prefactors, outp_feat.prefactors)
        self.regul = regul

class CovOp(FiniteOp[InpVecT, InpVecT]):
    def __init__(self, inp_feat:InpVecT, regul = 0.01, center = False):
        if center:
            inp_feat = inp_feat.centered()
        out_feat = inp_feat
        # if len(inp_feat.reduce) > 0 and isinstance(inp_feat.reduce[-1], Prefactors):
        #     matr = np.diag(inp_feat.reduce[-1].prefactors)
        #     reduce = copy(inp_feat.reduce)
        #     reduce[-1] = Prefactors(np.ones(len(inp_feat)))
        #     out_feat = FiniteVec(inp_feat.k, inp_feat.inspace_points, reduce)
        # else:
        #     matr = np.diag(np.ones(len(inp_feat)))
        super().__init__(inp_feat,
                         out_feat,
        #                 matr,
                         np.eye(len(inp_feat)))
        self.inp_feat = self.outp_feat
        self._inv = None
        self.regul = regul
    
    @classmethod
    def from_Samples(cls, kern, inspace_points, prefactors = None, regul = 0.01):
        return cls(FiniteVec(kern, inspace_points, prefactors), regul = regul)
    
    @classmethod
    def regul(cls, nsamps:int, nrefsamps:int = None, a:float = 0.49999999999999, b:float = 0.49999999999999, c:float = 0.1):
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

    def inv(self, regul:float = None) -> "CovOp[InpVecT, InpVecT]":
        """Compute the inverse of this covariance operator with a certain regularization.

        Args:
            regul (float, optional): Regularization parameter to be used. Defaults to self.regul.

        Returns:
            CovOp[InpVecT, InpVecT]: The inverse operator
        """
        set_inv = False
        rval = self._inv

        if regul is None:
            set_inv = True
            regul = self.regul        
        print("regul=", regul)
        if self._inv is None or set_inv:
            inv_gram = np.linalg.inv(inner(self.inp_feat) + regul * np.eye(len(self.inp_feat), dtype = self.matr.dtype))
            matr = (self.matr @ self.matr @ inv_gram @ inv_gram)
            rval = CovOp(self.inp_feat, regul)
            rval.matr = matr
            rval._inv = self
            
        if set_inv:
            self._inv = rval
        return rval
    
    def solve(self, inp:CombT) -> RkhsObject:
        """If `inp` is an RKHS vector of length 1 (a mean embedding): Solve the inverse problem to find dP/dρ from equation
        μ_P = C_ρ dP/dρ
        where C_ρ is the covariance operator represented by this object (`self`), ρ is the reference distribution, and μ_P is given by `inp`.
        If `inp` is a `FiniteOp`: Solve the inverse problem to find operator B from equation
        A = C_ρ B
        where C_ρ is the covariance operator represented by this object (`self`), and A is given by `inp`.
        
        Args:
            inp (InpVecT): The embedding of the distribution of interest, or the LinOp of interest.
        """
        
        if isinstance(inp, FiniteOp):
            reg_inp = inp.outp_feat
        else:
            if isinstance(inp, DeviceArray):
                inp = FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
            #assert(len(inp) == 1)
            reg_inp = inp

        regul = CovOp.regul(max(reg_inp.nsamps().min(), 1), max(self.inp_feat.nsamps(True), 1))
        return (self.inv(regul) @ inp)

    def eig(self) -> FiniteVec:
        G = self.inp_feat.inner()
        np.linalg.eigh(self.matr @ G + np.eye(G.shape[0])/1000 )
        
    

        

class Cmo(FiniteOp[InpVecT, OutVecT]):
    """conditional mean operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul:float = 0.01, center = False, regul_func = None):
        regul = np.array(regul, dtype=np.float32)

        #we do not center the output features - this still leads to the correct results in the output of the CME
        c_inp_feat = inp_feat.centered() if center else inp_feat
        G = inner(c_inp_feat)
        if regul_func is None:
            assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)       
            matr = np.linalg.inv(G + regul * np.eye(len(inp_feat)))
        else:
            matr = regul_func(G)
        super().__init__(inp_feat, outp_feat, matr, mean_center_inp=center, decenter_outp = center)

class Cdo(FiniteOp[InpVecT, OutVecT]):
    """conditional density operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, ref_feat:OutVecT, regul = 0.01, center:bool = False):
        #if center:
        #    outp_feat = outp_feat.extend_reduce([DecenterOutFeat(np.eye(len(outp_feat)))])
        mo = Cmo(inp_feat, outp_feat, regul, center = center)
        matr = CovOp(ref_feat, regul).solve(mo).matr
        super().__init__(mo.inp_feat,
                         ref_feat,
                         matr,
                         mean_center_inp = center, decenter_outp = False, normalize=True)


class HsTo(FiniteOp[InpVecT, InpVecT]): 
    """RKHS transfer operators
    """
    def __init__(self, start_feat:InpVecT, timelagged_feat:InpVecT, regul = 0.01, embedded = True, koopman = False):
        self.embedded = embedded
        self.koopman = koopman
        assert(start_feat.k == timelagged_feat.k)
        if (embedded is True and koopman is False) or (embedded is False and koopman is True):
            self.matr = np.linalg.inv(inner(start_feat) + timelagged_feat * regul * np.eye(len(start_feat)))
        else:
            G_xy = inner(start_feat, timelagged_feat)
            G_x = inner(start_feat)
            matr = (np.linalg.pinv(G_xy)
                         @ np.linalg.pinv(G_x+ len(timelagged_feat) * regul * np.eye(len(timelagged_feat))) 
                         @ G_xy)
            if koopman is True:
                matr = self.matr.T

        if koopman is False:
            inp_feat = start_feat
            outp_feat = timelagged_feat
        else:
            inp_feat = timelagged_feat
            outp_feat = start_feat
        super().__init__(inp_feat, outp_feat, matr)
