from .base import Kernel, DensityKernel
from .simple import FeatMapKernel, LinearKernel,  DictKernel
from .rbf import GenGaussKernel, GaussianKernel, LaplaceKernel, PeriodicKernel, ThreshSpikeKernel
from .adapt_combine import SplitDimsKernel, SKlKernel