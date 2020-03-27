import copy

import jax.numpy as np
from numpy.random import randn
import pytest
from numpy.testing import assert_allclose
from jax import random

from jaxrk.rkhs import CovOp, Cdo, Cmo, FiniteOp, FiniteVec, multiply, inner, SiEdSpVec, SpVec, CombVec
from jaxrk.kern import (GaussianKernel, SplitDimsKernel, PeriodicKernel)
from jaxrk.utilities.array_manipulation import all_combinations

rng = random.PRNGKey(1)



kernel_setups = [
    GaussianKernel()
]


def test_multiply():
    x = np.linspace(-2.5, 15, 5)[:, np.newaxis].astype(np.float32)
    y = randn(x.size)[:, np.newaxis].astype(np.float32)
    
    gk_x = GaussianKernel(0.1)

    
    x_e1 = FiniteVec.construct_RKHS_Elem(gk_x, x)
    x_e2 = FiniteVec.construct_RKHS_Elem(gk_x, y)
    x_fv = FiniteVec(gk_x, np.vstack([x,y]), prefactors = np.hstack([x_e1.prefactors] * 2), points_per_split=x.size)

    oper_feat_vec  = FiniteVec(gk_x, x)

    oper = FiniteOp(oper_feat_vec, oper_feat_vec, np.eye(len(x)))
    res_e1 = multiply(oper, x_e1)
    res_e2 = multiply(oper, x_e2)
    res_v = multiply(oper, x_fv)
    assert np.allclose(res_e1.prefactors, (oper.matr @ oper.inp_feat.inner(x_e1)).flatten()), "Application of operator to RKHS element failed."
    assert np.allclose(res_v.inspace_points, np.vstack([res_e1.inspace_points, res_e2.inspace_points] )), "Application of operator to all vectors in RKHS vector failed at inspace points."
    assert np.allclose(res_v.prefactors, np.hstack([res_e1.prefactors, res_e2.prefactors])), "Application of operator to all vectors in RKHS vector failed."
    assert np.allclose(multiply(oper, oper).matr, oper.inp_feat.inner(oper.outp_feat)), "Application of operator to operator failed."

def test_FiniteOp():
    gk_x = GaussianKernel(0.1)
    x = np.linspace(-2.5, 15, 20)[:, np.newaxis].astype(np.float)
    #x = np.random.randn(20, 1).astype(np.float)
    ref_fvec = FiniteVec(gk_x, x, np.ones(len(x)))
    ref_elem = FiniteVec.construct_RKHS_Elem(gk_x, x, np.ones(len(x)))

    C1 = FiniteOp(ref_fvec, ref_fvec, np.linalg.inv(inner(ref_fvec)))
    assert(np.allclose(multiply(C1, ref_elem).prefactors, 1.))

    C2 = FiniteOp(ref_fvec, ref_fvec, C1.matr@C1.matr)
    assert(np.allclose(multiply(C2, ref_elem).prefactors, np.sum(C1.matr, 0)))

    n_rvs = 50
    rv_fvec = FiniteVec(gk_x, np.random.randn(n_rvs).reshape((-1, 1)) * 5, np.ones(n_rvs))
    C3 = FiniteOp(rv_fvec, rv_fvec, np.eye(n_rvs))
    assert(np.allclose(multiply(C3, C1).matr, gk_x(rv_fvec.inspace_points, ref_fvec.inspace_points) @ C1.matr), 0.001, 0.001)


def test_CovOp(plot = False):   
    import distributions as dist

    nsamps = 1000
    samps_unif = None
    regul_C_ref=0.0001
    D = 1
    import pylab as pl
    if samps_unif is None:
        samps_unif = nsamps
    gk_x = GaussianKernel(0.2)

    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1]).astype(float)
    out_fvec = FiniteVec(gk_x, out_samps, np.ones(nsamps))

    
    #gk_x = LaplaceKernel(3)
    #gk_x = StudentKernel(0.7, 15)
    x = np.linspace(-2.5, 15, samps_unif)[:, np.newaxis].astype(float)
    ref_fvec = FiniteVec(gk_x, x, np.ones(len(x)))
    ref_elem = FiniteVec.sum()

    C_ref = CovOp(ref_fvec, regul=0.) # CovOp_compl(out_fvec.k, out_fvec.inspace_points, regul=0.)

    inv_Gram_ref = np.linalg.inv(inner(ref_fvec))
    assert(np.allclose((inv_Gram_ref@inv_Gram_ref)/ C_ref.inv().matr, 1., atol = 1e-3))
    #assert(np.allclose(multiply(C_ref.inv(), ref_elem).prefactors, np.sum(np.linalg.inv(inner(ref_fvec)), 0), rtol=1e-02))

    C_samps = CovOp(out_fvec, regul=regul_C_ref)
    unif_obj = multiply(C_samps.inv(), FiniteVec.construct_RKHS_Elem(out_fvec.kern, out_fvec.inspace_points, out_fvec.prefactors).normalized())
    C_ref = CovOp(ref_fvec, regul=regul_C_ref)
    dens_obj = multiply(C_ref.inv(), FiniteVec.construct_RKHS_Elem(out_fvec.kern, out_fvec.inspace_points, out_fvec.prefactors)).normalized()
    


    #dens_obj.prefactors = np.sum(dens_obj.prefactors, 1)
    #dens_obj.prefactors = dens_obj.prefactors / np.sum(dens_obj.prefactors)
    #print(np.sum(dens_obj.prefactors))
    #p = np.sum(inner(dens_obj, ref_fvec), 1)
    targp = np.exp(targ.logpdf(ref_fvec.inspace_points.squeeze())).squeeze()
    estp = np.squeeze(inner(dens_obj, ref_fvec))
    estp2 = np.squeeze(inner(dens_obj.unsigned_projection().normalized(), ref_fvec))
    assert(np.abs(targp.squeeze()-estp).mean() < 0.8)
    if plot:
        pl.plot(ref_fvec.inspace_points.squeeze(), estp/np.max(estp) * np.max(targp), "b--", label="scaled estimate")
        pl.plot(ref_fvec.inspace_points.squeeze(), estp2/np.max(estp2) * np.max(targp), "g-.", label="scaled estimate (uns)")
        pl.plot(ref_fvec.inspace_points.squeeze(), targp, label = "truth")
        
        
        #pl.plot(ref_fvec.inspace_points.squeeze(), np.squeeze(inner(unif_obj, ref_fvec)), label="unif")
        pl.legend(loc="best")
        pl.show()
    assert(np.std(np.squeeze(inner(unif_obj.normalized(), out_fvec))) < 0.1)




def test_Cdmo(plot = False):
    

    def generate_donut(nmeans = 10, nsamps_per_mean = 50):
        from scipy.stats import multivariate_normal
        import distributions as dist
        from numpy import exp

        def pol2cart(theta, rho):
            x = (rho * np.cos(theta)).reshape(-1,1)
            y = (rho * np.sin(theta)).reshape(-1,1)
            return np.concatenate([x, y], axis = 1)

        comp_distribution = multivariate_normal(np.zeros(2), np.eye(2)/100)
        means = pol2cart(np.linspace(0,2*3.141, nmeans + 1)[:-1], 1)

        rvs = comp_distribution.rvs(nmeans * nsamps_per_mean) + np.repeat(means, nsamps_per_mean, 0)
        true_dens = lambda samps: exp(dist.mixture.location_mixture_logpdf(samps, means, np.ones(nmeans) / nmeans, comp_distribution))
        return rvs, means, true_dens

    x1 = np.ones((1,1))
    x2 = np.zeros((1,1))
    (rvs, means, true_dens) = generate_donut(50, 10)
    invec = FiniteVec(GaussianKernel(0.5), rvs[:, :1])
    outvec = FiniteVec(GaussianKernel(0.5), rvs[:, 1:])
    refervec = FiniteVec(outvec.k, np.linspace(-4, 4, 5000)[:, None])
    cd = Cdo(invec, outvec, refervec, 0.1)
    cm =  Cmo(invec, outvec, 0.1)
    (true_x1, est_x1, este_x1, true_x2, est_x2, este_x2) = [lambda samps: true_dens(np.hstack([np.repeat(x1, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(multiply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(multiply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: true_dens(np.hstack([np.repeat(x2, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(multiply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(multiply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps)))))]

    t = np.array((true_x1(refervec.inspace_points), true_x2(refervec.inspace_points)))
    e = np.array((est_x1(refervec.inspace_points), est_x2(refervec.inspace_points)))
    if plot:
        import pylab as pl

        (fig, ax) = pl.subplots(1, 3, False, False)
        ax[0].plot(refervec.inspace_points, t[0])
        ax[0].plot(refervec.inspace_points, e[0], "--", label = "dens")
        ax[0].plot(refervec.inspace_points, este_x1(refervec.inspace_points), "-.", label = "emb")
        
        ax[1].plot(refervec.inspace_points, t[1])
        ax[1].plot(refervec.inspace_points, e[1], "--", label = "dens")
        ax[1].plot(refervec.inspace_points, este_x2(refervec.inspace_points),"-.", label = "emb")

        ax[2].scatter(*rvs.T)
        fig.legend()
        fig.show()
    assert(np.allclose(e,t, atol=0.5))


def test_Cdo_timeseries(plot = False):
    if plot:
        import pylab as pl
    x = np.linspace(0, 40, 400).reshape((-1, 1))
    y = np.sin(x) + randn(len(x)).reshape((-1, 1)) * 0.2
    proc_data = np.hstack([x,y])
    if plot:
        pl.plot(x.flatten(), y.flatten())

    invec = FiniteVec(GaussianKernel(0.5), np.array([y.squeeze()[i:i+10] for i in range(190)]))
    outvec = FiniteVec(GaussianKernel(0.5), y[10:200])
    refervec = FiniteVec(outvec.k, np.linspace(y[:-201].min() - 2, y[:-201].max() + 2, 5000)[:, None])
    cd = Cdo(invec, outvec, refervec, 0.1)
    cd = Cmo(invec, outvec, 0.1)
    sol2 = np.array([multiply(cd, FiniteVec(invec.k, y[end-10:end].T)).normalized().get_mean_var() for end in range(200,400) ])
    if plot:
        pl.plot(x[200:].flatten(), sol2.T[0].flatten())
    invec = CombVec(FiniteVec(PeriodicKernel(np.pi, 5), x[:200,:]),
                     SpVec(SplitDimsKernel([0,1,2],[PeriodicKernel(np.pi, 5), GaussianKernel(0.1)]),
                           proc_data[:200,:], np.array([200]), use_subtrajectories=True), np.multiply)
    outvec = FiniteVec(GaussianKernel(0.5), y[1:-199])
    #cd = Cdo(invec, outvec, refervec, 0.1)
    cd = Cmo(invec, outvec, 0.1)
    #sol = (cd.inp_feat.inner(SpVec(invec.k, proc_data[:230], np.array([230]), use_subtrajectories=True)))
    #sol = [(cd.inp_feat.inner(SiEdSpVec(invec.k_obs, y[:end], np.array([end]), invec.k_idx, use_subtrajectories=False ))) for end in range(200,400) ]
    #pl.plot(np.array([sol[i][-1] for i in range(len(sol))]))

    #sol = np.array([multiply (cd, SpVec(invec.k, proc_data[:end], np.array([end]), use_subtrajectories=False)).normalized().get_mean_var() for end in range(200,400) ])
    sol = multiply (cd, CombVec(FiniteVec(invec.v1.k, x), SpVec(invec.v2.k, proc_data[:400], np.array([400]), use_subtrajectories=True), np.multiply)).normalized().get_mean_var()


    print(sol)
    return sol2.T[0], sol.T[0][200:], y[200:]
    (true_x1, est_x1, este_x1, true_x2, est_x2, este_x2) = [lambda samps: true_dens(np.hstack([np.repeat(x1, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(multiply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(multiply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: true_dens(np.hstack([np.repeat(x2, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(multiply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(multiply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps)))))]

    t = np.array((true_x1(refervec.inspace_points), true_x2(refervec.inspace_points)))
    e = np.array((est_x1(refervec.inspace_points), est_x2(refervec.inspace_points)))
    if plot:
        import pylab as pl

        (fig, ax) = pl.subplots(1, 3, False, False)
        ax[0].plot(refervec.inspace_points, t[0])
        ax[0].plot(refervec.inspace_points, e[0], "--", label = "dens")
        ax[0].plot(refervec.inspace_points, este_x1(refervec.inspace_points), "-.", label = "emb")
        
        ax[1].plot(refervec.inspace_points, t[1])
        ax[1].plot(refervec.inspace_points, e[1], "--", label = "dens")
        ax[1].plot(refervec.inspace_points, este_x2(refervec.inspace_points),"-.", label = "emb")

        ax[2].scatter(*rvs.T)
        fig.legend()
        fig.show()
    assert(np.allclose(e,t, atol=0.5))
