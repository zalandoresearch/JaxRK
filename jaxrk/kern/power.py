#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:41:58 2019

@author: Ingmar Schuster
"""
#%%
import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.base import RKHSOperator
from rkhsop.op.cond_probability import ConditionDensityOperator, ConditionMeanEmbedding
from rkhsop.op.preimg_densest import IsPreimgOp, RmPreimgOp
import rkhsop.kern.base as kern
from rkhsop.experiments.data import Power

import pylab as pl

def run_power(idx = 0):
    d = Power().get_ihgp_window(idx)['Global_active_power'].values.reshape(-1, 1)
    L_samp = np.linspace(d.min()-0.1, d.max()+0.1, 500).reshape(-1, 1)
    inp = np.arange(len(d)).reshape(-1, 1)
    print(d.shape, L_samp.shape, inp.shape)
    cdo = ConditionDensityOperator(inp, d, L_samp,
                                            kern.GaussianKernel(1), kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1])),
                                            inp_regul=0.001, outp_regul=0.0001)
    inp_fine = np.linspace(d.min()-0.1, d.max()+0.1, 800).reshape(-1, 1)
    (m, v) = cdo.mean_var(inp_fine)
    sd = np.sqrt(v)
    pl.scatter(inp, d, alpha = 0.2)
    pl.plot(inp_fine[1:], m, "b--", alpha = 0.5)
    pl.fill_between(inp_fine[1:].flatten(), m + 2 * sd, m - 2 * sd, color='r', alpha=0.2)

    

