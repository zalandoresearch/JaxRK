import jax.numpy as np

from typing import TypeVar,  Generic
from jaxrk.rkhs.vector import FiniteVec, inner

from .base import Op, Vec, RkhsObject
from scipy.optimize import minimize

#VecSubType = NewType("VecSubType", Vec)
InpVecT = TypeVar("InpVecT", bound=Vec)
IntermVecT = TypeVar("IntermVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)



class FiniteOp(Op[InpVecT, OutVecT]):
    """Finite rank RKHS operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, matr:np.array):
        self.inp_feat = inp_feat
        self.outp_feat = outp_feat
        self.matr = matr
    
    def __len__(self):
        return len(self.inp_feat)
    
    def solve(self, result:FiniteVec):
        if np.all(self.outp_feat.inspace_points == result.inspace_points):
            s = np.linalg.solve(self.matr @ inner(self.inp_feat, self.inp_feat), result.prefactors)
            return FiniteVec.construct_RKHS_Elem(result.k, result.inspace_points, s)
        else:
            assert()



class CrossCovOp(FiniteOp[InpVecT, OutVecT]):
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul = 0.01):
        assert len(inp_feat) == len(outp_feat)
        assert np.allclose(inp_feat.prefactors, outp_feat.prefactors)
        self.inp_feat = inp_feat
        self.outp_feat = outp_feat
        self.matr = np.diag((inp_feat.prefactors + outp_feat.prefactors)/2)
        self.regul = regul

class CovOp(FiniteOp[InpVecT, InpVecT]):
    def __init__(self, inp_feat:InpVecT, regul = 0.01):
        self.inp_feat = self.outp_feat = self.inp_feat = inp_feat.updated(np.ones(len(inp_feat),dtype = inp_feat.prefactors.dtype))
        self.matr = np.diag(inp_feat.prefactors)
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
    
    def solve(self, embedding:InpVecT):
        """Solve the inverse problem to find
        μ_P = C_ρ dP/dρ
        where C_ρ is the covariance operator represented by this object (`self`), ρ is the reference distribution, and μ_P is given by `embedding`.

        Args:
            embedding (InpVecT): The embedding of the distribution of interest.
        """
        regul = CovOp.regul(1./np.mean(embedding.prefactors), 1./np.mean(np.diag(self.matr)))
        C_inv =  self.inv(regul)
        return multiply(C_inv, embedding).normalized()

        

class Cmo(FiniteOp[InpVecT, OutVecT]):
    """conditional mean operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, regul:float = 0.01):
        self.inp_feat = inp_feat
        self.outp_feat = outp_feat
        regul = np.array(regul, dtype=np.float32)
        if False:
            op = multiply(CrossCovOp(inp_feat, outp_feat), CovOp(inp_feat, regul).inv())
            (self.inp_feat, self.outp_feat, self.matr) = (op.inp_feat,
                                                          op.outp_feat,
                                                          op.matr)
        else:
            self.matr = np.linalg.inv(inner(self.inp_feat) + regul * np.eye(len(inp_feat)))

class Cdo(FiniteOp[InpVecT, OutVecT]):
    """conditional density operator
    """
    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, ref_feat:OutVecT, regul = 0.01):
        
        if True:
            op = multiply(CovOp(ref_feat, regul).inv(), Cmo(inp_feat, outp_feat, regul))
            (self.inp_feat, self.outp_feat, self.matr) = (op.inp_feat,
                                                          op.outp_feat,
                                                          op.matr)
        else:
            self.inp_feat = inp_feat
            self.outp_feat = ref_feat
            cmo_matr = np.linalg.inv(inner(self.inp_feat) + regul * tf.eye(len(inp_feat)))
            assert np.allclose(cmo_matr, Cmo(inp_feat, outp_feat, regul).matr)

            inv_gram = np.linalg.inv(inner(ref_feat) + regul * np.eye(len(ref_feat), dtype = ref_feat.prefactors.dtype))
            preimg_factor = (np.diag(ref_feat.prefactors**2) @ inv_gram @ inv_gram)
            #assert np.allclose(preimg_factor, CovOp(ref_feat, regul).inv().matr)
            self.matr =  preimg_factor @ inner(ref_feat, outp_feat) @ cmo_matr


class HsTo(FiniteOp[InpVecT, InpVecT]): 
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
            self.matr = (np.linalg.pinv(G_xy)
                         @ np.linalg.pinv(G_x+ len(timelagged_feat) * regul * np.eye(len(timelagged_feat))) 
                         @ G_xy)
            if koopman is True:
                self.matr = self.matr.T

        if koopman is False:
            self.inp_feat = start_feat
            self.outp_feat = timelagged_feat
        else:
            self.inp_feat = timelagged_feat
            self.outp_feat = start_feat


CombT = TypeVar("CombT", FiniteOp[InpVecT, IntermVecT], IntermVecT)

def multiply(A:FiniteOp[IntermVecT, OutVecT], B:CombT) -> RkhsObject: # "T = TypeVar("T"); multiply(A:FiniteOp, B:T) -> T"
    if isinstance(B, FiniteOp):
        return FiniteOp(B.inp_feat, A.outp_feat, A.matr @ inner(A.inp_feat, B.outp_feat) @ B.matr)
    else:
        if len(B) == 1:
            return FiniteVec.construct_RKHS_Elem(A.outp_feat.k, A.outp_feat.inspace_points, np.squeeze(A.matr @ inner(A.inp_feat, B)))
        else:
            pref = A.matr @ inner(A.inp_feat, B)
            return FiniteVec(A.outp_feat.k, np.tile(A.outp_feat.inspace_points, (pref.shape[1], 1)), np.hstack(pref.T), points_per_split=pref.shape[0])