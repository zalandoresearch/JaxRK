#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:55:59 2019

"""

import numpy as np

def all_combinations(data_a, data_b):
    """
    If we have a gram matrix G_a for data_a and a gram matrix G_b for data_b, then we can compute the gram matrix for np.stack(data_a, data_b, 1) as np.kron(G_a, G_b).
    To construct the dataset with blown up number of instances, use this function.
    """
    assert(len(data_a.shape) == 2)
    assert(len(data_b.shape) == 2)

    a = np.repeat(data_a, data_b.shape[0], axis = 0)
    b = np.tile(data_b, (data_a.shape[0], 1))
    #print(a, a.shape, b, b.shape)
    return np.hstack([a,b])
    