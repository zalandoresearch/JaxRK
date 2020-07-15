from typing import Generic, TypeVar, Callable

import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from jaxrk.rkhs.vector import FiniteVec, inner

from .base import Map, RkhsObject, Vec

InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

#The following is input to a map RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec) 

CombT = TypeVar("CombT", "FiniteMap[RhInpVectT, InpVecT]", InpVecT, np.array)


class FiniteMap(Map[InpVecT, OutVecT]):
    """Finite rank affine map in RKHS
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, matr:np.array, mean_center_inp:bool = False):
        self.inp_feat = inp_feat
        self.outp_feat = outp_feat
        self.matr = matr
        self.mean_center_inp = mean_center_inp
        if self.mean_center_inp:
            # compute the centering term that is constant wrt input
            evaluated_mean_emb = np.mean(inner(inp_feat, inp_feat), 1, keepdims = True)
            self.const_cent_term = np.mean(evaluated_mean_emb) - evaluated_mean_emb

    
    def __len__(self):
        return len(self.inp_feat)

    def _corrected_gram(self, input:InpVecT):
        inp_gram = inner(self.inp_feat, input)
        if self.mean_center_inp:
            inp_gram = (inp_gram 
                        + self.const_cent_term
                        - np.mean(inp_gram, 0, keepdims = True))
        return inp_gram

    def apply(self, inp:CombT) -> RkhsObject:
        if isinstance(inp, FiniteMap):
            return FiniteMap(inp.inp_feat, self.outp_feat, self.matr @ self._corrected_gram(inp.outp_feat) @ inp.matr, mean_center_inp = inp.mean_center_inp)
        else:
            if isinstance(inp, DeviceArray):
                inp = FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
            if len(inp) == 1:
                return FiniteVec.construct_RKHS_Elem(self.outp_feat.k, self.outp_feat.inspace_points, np.squeeze(self.matr @ self._corrected_gram(inp)))
            else:
                pref = self.matr @ self._corrected_gram(inp)
                return FiniteVec(self.outp_feat.k, np.tile(self.outp_feat.inspace_points, (pref.shape[1], 1)), np.hstack(pref.T), points_per_split=pref.shape[0])

    def __call__(self, inp:CombT) -> RkhsObject:
        return self.apply(inp)
    
    def solve(self, result:OutVecT):
        raise NotImplementedError()



class CrossCovOp(FiniteMap[InpVecT, OutVecT]):
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul = 0.01):
        super().__init__(inp_feat, outp_feat, np.diag((inp_feat.prefactors + outp_feat.prefactors)/2))
        assert len(inp_feat) == len(outp_feat)
        assert np.allclose(inp_feat.prefactors, outp_feat.prefactors)
        self.regul = regul

class CovOp(FiniteMap[InpVecT, InpVecT]):
    def __init__(self, inp_feat:InpVecT, regul = 0.01,center = False):
        super().__init__(inp_feat,
                         inp_feat.updated(np.ones(len(inp_feat),
                         dtype = inp_feat.prefactors.dtype)),
                         np.diag(inp_feat.prefactors),
                         mean_center_inp=center)
        self.inp_feat = self.outp_feat
        self._inv = None
        self.regul = regul
    
    @classmethod
    def from_Samples(cls, kern, inspace_points, prefactors = None, regul = 0.01):
        return cls(FiniteVec(kern, inspace_points, prefactors), regul = regul)
    
    @classmethod
    def regul(cls, nsamps:int, nrefsamps:int = None, a:float = 0.49999999999999, b:float = 0.49999999999999, c:float = 0.1):
        """Compute the regularizer based on the formula from the Kernel Conditional Density operators paper(Corollary 3.4, 2020).
        
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
        if regul is None:
            set_inv = True
            regul = self.regul

        if self._inv is None:
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
        If `inp` is a `FiniteMap`: Solve the inverse problem to find operator B from equation
        A = C_ρ B
        where C_ρ is the covariance operator represented by this object (`self`), and A is given by `inp`.
        
        Args:
            inp (InpVecT): The embedding of the distribution of interest, or the map of interest.
        """
        
        if isinstance(inp, FiniteMap):
            regul = CovOp.regul(len(inp.outp_feat), 1./np.mean(np.diag(self.matr)))
        else:
            if isinstance(inp, DeviceArray):
                inp = FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
            assert(len(inp) == 1)
            #regul = CovOp.regul(1./np.mean(embedding.prefactors), 1./np.mean(np.diag(self.matr)))
            regul = CovOp.regul(len(inp), 1./np.mean(np.diag(self.matr)))
        
        C_inv =  self.inv(regul)
        return apply(C_inv, inp)

        

class Cmo(FiniteMap[InpVecT, OutVecT]):
    """conditional mean operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul:float = 0.01, center = False):
        regul = np.array(regul, dtype=np.float32)

        #we do not center the output features - this still leads to the correct results in the output of the CME
        c_inp_feat = inp_feat.centered() if center else inp_feat            
        matr = np.linalg.inv(inner(c_inp_feat) + regul * np.eye(len(inp_feat)))
        super().__init__(inp_feat, outp_feat, matr, mean_center_inp=center)

class Cdo(FiniteMap[InpVecT, OutVecT]):
    """conditional density operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, ref_feat:OutVecT, regul = 0.01, center:bool = False):
        super().__init__(inp_feat,
                         ref_feat,
                         CovOp(ref_feat, regul).solve(Cmo(inp_feat, outp_feat, regul, center = center)).matr,
                         mean_center_inp = center)


class HsTo(FiniteMap[InpVecT, InpVecT]): 
    """RKHS transfer operators
    """
    def __init__(self, start_feat:InpVecT, timelagged_feat:InpVecT, regul = 0.01, embedded = False, koopman = False):
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




def apply(A:FiniteMap[InpVecT, OutVecT], inp:CombT) -> RkhsObject:
    return A.apply(inp)
