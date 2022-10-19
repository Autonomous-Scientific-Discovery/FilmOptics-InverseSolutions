# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:09:50 2021

@author: Rajdeep Dutta
"""

import numpy as np
from scipy import special


def gauss_mix(x, *gparams):
    g_count = len(gparams)/3
    def gauss_impl(x, A, mu, sigma):
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    res = np.zeros(len(x))
    for gi in range(int(g_count)):
        res += gauss_impl(x, gparams[gi*3], gparams[gi*3+1], gparams[gi*3+2])
    return res

def gausserfi_mix(x, *gparams):
    g_count = len(gparams)/3
    def gauss_impl(x, A, mu, sigma):
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    res = np.zeros(len(x))
    for gi in range(int(g_count)):
        res += gauss_impl(x, gparams[gi*3], gparams[gi*3+1], gparams[gi*3+2])*special.erfi(-(x-gparams[gi*3+1])/(np.sqrt(2)*gparams[gi*3+2]))
    return res