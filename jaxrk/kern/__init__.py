from .base import Kernel, DensityKernel, median_heuristic
from .simple import FeatMapKernel, LinearKernel, PeriodicKernel, ThreshSpikeKernel, DictKernel
from .rbf import GenGaussKernel,  GaussianKernel, LaplaceKernel
from .adapt_combine import SplitDimsKernel, SKlKernel