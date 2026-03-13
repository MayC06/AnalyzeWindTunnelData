# -*- coding: utf-8 -*-
"""
Created on Sat May 24 08:22:24 2025

Utils for pybounds mpc simulations.

@author: mayc06
"""

import numpy as np
import copy

def wrapTo2Pi(rad):
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap

def unwrap_angle(z, correction_window_for_2pi=100, n_range=2):
        
    smooth_zs = np.array(z[0:2])
    for i in range(2, len(z)):
        first_ix = np.max([0, i-correction_window_for_2pi])
        last_ix = i

        nbase = np.round( (smooth_zs[-1] - z[i])/(2*np.pi) )

        candidates = []
        for n in range(-1*n_range, n_range):
            candidates.append(n*2*np.pi+nbase*2*np.pi+z[i])
        error = np.abs(candidates - np.mean(smooth_zs[first_ix:last_ix])) 
        smooth_zs = np.hstack(( smooth_zs, [candidates[np.argmin(error)]] ))
    # if plot:
    #     plt.plot(smooth_zs, '.', color='black', markersize=1)
        
    return smooth_zs

def circular_diff(a, b, period=8):
    """
    Calculate the directional circular difference between values in two arrays
    a and b.

    Period is the total linear distance of the circular values. For example,
    the default period of 8 represents the 8 columns of the FB, and should
    produce difference values from -4 to 4, non-inclusive, because the maximum
    distance away any two points on the circle can be is 4 columns.
    """
    diff = np.mod(b - a, period)
    return np.where(diff > period / 2, diff - period, diff)