from jaxrk.utilities.eucldist import eucldist
from scipy.spatial.distance import cdist
from numpy.random import randn
from jax.numpy import allclose

def test_eucldist():
    a, b = randn(100, 3), randn(200,3)

    ab_cd = cdist(a, b, "sqeuclidean")
    ab_v1, ab_v2 = eucldist(a, b, power=2, variant="simple"), eucldist(a, b, power=2, variant="extension")

    aa_cd = cdist(a, a, "sqeuclidean")
    aa_v1, aa_v2 = eucldist(a, power=2, variant="simple"), eucldist(a, power=2, variant="extension")

    for (ground_truth, est) in [(ab_cd, [ab_v1, ab_v2]),
                                (aa_cd, [aa_v1, aa_v2])]:
        for variant in est:
            assert(allclose(ground_truth, variant, atol = 1e-05))