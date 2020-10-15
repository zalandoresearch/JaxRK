from .base import Kernel, DensityKernel, median_heuristic
from .simple import FeatMapKernel, LinearKernel, PeriodicKernel, SpikeKernel
from .rbf import GenGaussKernel,  GaussianKernel, LaplaceKernel
from .adapt_combine import SplitDimsKernel, SKlKernel