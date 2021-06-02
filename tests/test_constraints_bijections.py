import jax.numpy as np
from jaxrk.core.constraints import SoftBd

def test_simple_bijections(atol = 1e-3):
    for lb in (-0.4, 5.):
        for ub in (5.5, 6.):
            assert lb < ub
            l = SoftBd(lower_bound=lb)
            u = SoftBd(upper_bound=ub)
            lu = SoftBd(lb, ub)

            x = np.linspace(lb - 0.01, ub + 0.01, 1000)
            lx = l(x)
            ux = u(x)
            lux = lu(x)
            assert np.all(lx > lb)
            assert np.all(ux < ub)
            assert np.all(lux > lb) and np.all(lux < ub)
            for i, n in [(l.inv(lx), "lower"), (lu.inv(lux), "lower & upper"), (u.inv(ux), "upper")]:
                assert np.allclose(x, i, atol = atol), f"Inverse of {n} bound Bijection is inaccurate(max abs error: {np.abs(x-i).max()}, mean abs error: {np.abs(x-i).mean()}, min abs error: {np.abs(x-i).min()})"