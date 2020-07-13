import copy

import jax.numpy as np
from numpy.random import randn
import pytest
from numpy.testing import assert_allclose
from jax import random

from jaxrk.rkhs import CovOp, Cdo, Cmo, FiniteMap, FiniteVec, apply, inner, SpVec, CombVec
from jaxrk.kern import (GaussianKernel, SplitDimsKernel, PeriodicKernel)
from jaxrk.utilities.array_manipulation import all_combinations

from mixtures_tools import location_mixture_logpdf, mixt

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

    oper = FiniteMap(oper_feat_vec, oper_feat_vec, np.eye(len(x)))
    res_e1 = apply(oper, x_e1)
    res_e2 = apply(oper, x_e2)
    res_v = apply(oper, x_fv)
    assert np.allclose(res_e1.prefactors, (oper.matr @ oper.inp_feat.inner(x_e1)).flatten()), "Application of operator to RKHS element failed."
    assert np.allclose(res_v.inspace_points, np.vstack([res_e1.inspace_points, res_e2.inspace_points] )), "Application of operator to all vectors in RKHS vector failed at inspace points."
    assert np.allclose(res_v.prefactors, np.hstack([res_e1.prefactors, res_e2.prefactors])), "Application of operator to all vectors in RKHS vector failed."
    assert np.allclose(apply(oper, oper).matr, oper.inp_feat.inner(oper.outp_feat)), "Application of operator to operator failed."

def test_FiniteMap():
    gk_x = GaussianKernel(0.1)
    x = np.linspace(-2.5, 15, 20)[:, np.newaxis].astype(np.float32)
    #x = np.random.randn(20, 1).astype(np.float)
    ref_fvec = FiniteVec(gk_x, x, np.ones(len(x)))
    ref_elem = FiniteVec.construct_RKHS_Elem(gk_x, x, np.ones(len(x)))

    C1 = FiniteMap(ref_fvec, ref_fvec, np.linalg.inv(inner(ref_fvec)))
    assert(np.allclose(apply(C1, ref_elem).prefactors, 1.))

    C2 = FiniteMap(ref_fvec, ref_fvec, C1.matr@C1.matr)
    assert(np.allclose(apply(C2, ref_elem).prefactors, np.sum(C1.matr, 0)))

    n_rvs = 50
    rv_fvec = FiniteVec(gk_x, random.normal(rng, (n_rvs, 1)) * 5, np.ones(n_rvs))
    C3 = FiniteMap(rv_fvec, rv_fvec, np.eye(n_rvs))
    assert np.allclose(apply(C3, C1).matr, gk_x(rv_fvec.inspace_points, ref_fvec.inspace_points) @ C1.matr, 0.001, 0.001)


def test_CovOp(plot = False):   
    from scipy.stats import multivariate_normal

    nsamps = 1000
    samps_unif = None
    regul_C_ref=0.0001
    D = 1
    import pylab as pl
    if samps_unif is None:
        samps_unif = nsamps
    gk_x = GaussianKernel(0.2)

    targ = mixt(D, [multivariate_normal(3*np.ones(D), np.eye(D)*0.7**2), multivariate_normal(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1]).astype(float)
    out_fvec = FiniteVec(gk_x, out_samps, np.ones(nsamps))
    out_meanemb = out_fvec.sum()
    

    x = np.linspace(-2.5, 15, samps_unif)[:, np.newaxis].astype(float)
    ref_fvec = FiniteVec(gk_x, x, np.ones(len(x)))
    ref_elem = ref_fvec.sum()

    C_ref = CovOp(ref_fvec, regul=0.) # CovOp_compl(out_fvec.k, out_fvec.inspace_points, regul=0.)

    inv_Gram_ref = np.linalg.inv(inner(ref_fvec))

    C_samps = CovOp(out_fvec, regul=regul_C_ref)
    unif_obj = C_samps.solve(out_meanemb).unsigned_projection().normalized()
    C_ref = CovOp(ref_fvec, regul=regul_C_ref)
    dens_obj = C_ref.solve(out_meanemb).unsigned_projection().normalized()
    


    targp = np.exp(targ.logpdf(ref_fvec.inspace_points.squeeze())).squeeze()
    estp = np.squeeze(inner(dens_obj, ref_fvec))
    estp2 = np.squeeze(inner(dens_obj, ref_fvec))
    est_sup = unif_obj(x).squeeze()
    assert (np.abs(targp.squeeze()-estp).mean() < 0.8), "Estimated density strongly deviates from true density"
    if plot:
        pl.plot(ref_fvec.inspace_points.squeeze(), estp/np.max(estp) * np.max(targp), "b--", label="scaled estimate")
        pl.plot(ref_fvec.inspace_points.squeeze(), estp2/np.max(estp2) * np.max(targp), "g-.", label="scaled estimate (uns)")
        pl.plot(ref_fvec.inspace_points.squeeze(), targp, label = "truth")
        pl.plot(x.squeeze(), est_sup.squeeze(), label = "support")
        
        #pl.plot(ref_fvec.inspace_points.squeeze(), np.squeeze(inner(unif_obj, ref_fvec)), label="unif")
        pl.legend(loc="best")
        pl.show()
    supp = unif_obj(x).squeeze()
    assert (np.std(supp) < 0.15), "Estimated support has high variance, in data points, while it should be almost constant."




def test_Cdmo(plot = False):
    

    def generate_donut(nmeans = 10, nsamps_per_mean = 50):
        from scipy.stats import multivariate_normal
        from numpy import exp

        def pol2cart(theta, rho):
            x = (rho * np.cos(theta)).reshape(-1,1)
            y = (rho * np.sin(theta)).reshape(-1,1)
            return np.concatenate([x, y], axis = 1)

        comp_distribution = multivariate_normal(np.zeros(2), np.eye(2)/100)
        means = pol2cart(np.linspace(0,2*3.141, nmeans + 1)[:-1], 1)

        rvs = comp_distribution.rvs(nmeans * nsamps_per_mean) + np.repeat(means, nsamps_per_mean, 0)
        true_dens = lambda samps: exp(location_mixture_logpdf(samps, means, np.ones(nmeans) / nmeans, comp_distribution))
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
                                                            apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(),
                                                            apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(),
                                                            lambda samps: true_dens(np.hstack([np.repeat(x2, len(samps), 0), samps])),
                                                            apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(),
                                                            apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), ]
                                                            # lambda samps: np.squeeze(inner(apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            # lambda samps: np.squeeze(inner(apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            # lambda samps: true_dens(np.hstack([np.repeat(x2, len(samps), 0), samps])),
                                                            # lambda samps: np.squeeze(inner(apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            # lambda samps: np.squeeze(inner(apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps)))))]

    t = np.array((true_x1(refervec.inspace_points), true_x2(refervec.inspace_points)))
    e = np.array((np.squeeze(est_x1(refervec.inspace_points)), np.squeeze(est_x2(refervec.inspace_points))))
    if plot:
        import pylab as pl

        (fig, ax) = pl.subplots(1, 3, False, False)
        ax[0].plot(refervec.inspace_points, t[0])
        ax[0].plot(refervec.inspace_points, e[0], "--", label = "dens")
        ax[0].plot(refervec.inspace_points, np.squeeze(este_x1(refervec.inspace_points)), "-.", label = "emb")
        
        ax[1].plot(refervec.inspace_points, t[1])
        ax[1].plot(refervec.inspace_points, e[1], "--", label = "dens")
        ax[1].plot(refervec.inspace_points, np.squeeze(este_x2(refervec.inspace_points)),"-.", label = "emb")

        ax[2].scatter(*rvs.T)
        fig.legend()
        fig.show()
    assert(np.allclose(e,t, atol=0.5))


def test_Cdo_timeseries(plot = False):
    raise NotImplementedError()
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
    sol2 = np.array([apply(cd, FiniteVec(invec.k, y[end-10:end].T)).normalized().get_mean_var() for end in range(200,400) ])
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
    sol = apply(cd, CombVec(FiniteVec(invec.v1.k, x), SpVec(invec.v2.k, proc_data[:400], np.array([400]), use_subtrajectories=True), np.multiply)).normalized().get_mean_var()


    print(sol)
    return sol2.T[0], sol[0][200:], y[200:]
    (true_x1, est_x1, este_x1, true_x2, est_x2, este_x2) = [lambda samps: true_dens(np.hstack([np.repeat(x1, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x1)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: true_dens(np.hstack([np.repeat(x2, len(samps), 0), samps])),
                                                            lambda samps: np.squeeze(inner(apply(cd, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps))))),
                                                            lambda samps: np.squeeze(inner(apply(cm, FiniteVec.construct_RKHS_Elem(invec.k, x2)).normalized().unsigned_projection().normalized(), FiniteVec(refervec.k, samps, prefactors=np.ones(len(samps)))))]

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
