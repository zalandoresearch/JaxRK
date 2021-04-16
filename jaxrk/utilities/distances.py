import jax.numpy as np
from typing import TypeVar
from ..core.typing import Array
from ..rkhs.base import Vec
from .gram import rkhs_gram_cdist
from .eucldist import eucldist


__all__ = ["dist", "median_heuristic"]

def median_heuristic(data, distance, per_dimension = True):
    import numpy as onp
    from scipy.spatial.distance import pdist
    if isinstance(distance, str):
        dist_fn = lambda x: pdist(x, distance)
    else:
        dist_fn = distance
    if per_dimension is False:
        return onp.median(dist_fn(data))
    else:
        def single_dim_heuristic(data_dim):
            return median_heuristic(data_dim[:, None], dist_fn, per_dimension = False)
        return onp.apply_along_axis(single_dim_heuristic, 0, data)


T = TypeVar('T', Vec, Array)

def rkhs_cdist(a:Vec, b:Vec = None, power:float = 2.):
    """Compute RKHS distances between RKHS elements in vectors a and b

    Args:
        a (Vec): Vector to compute distances from.
        b (Vec, optional): Vector to compute distances to. Defaults to None, in which case the function returns the distances between all elements of a.
        power (float, optional): The power to raise the distance to. Defaults to 2.

    Returns:
        [type]: [description]
    """
    if a == b or b is None:
        return rkhs_gram_cdist(a.inner(), power = power)
    else:
        return rkhs_gram_cdist(a.inner(b), a.inner(), b.inner(), power = power)

def dist(a:T, b:T = None, power:float = 2.):
    dfunc = rkhs_cdist if isinstance(a, Vec) else eucldist
    return dfunc(a, b, power = power)

