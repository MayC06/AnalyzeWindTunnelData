# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:20:43 2025

@author: mayc06
"""
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pynumdiff
import utils
from utils import unwrap_angle,wrapToPi

import warnings
from scipy import special


def get_trajlist_from_behdata(csv,trajstart,trajstop,windspeed=40):
    
    '''
    A function to easily grab a list of trajectories from behavior data 
    from the van Breugel lab wind tunnels (Stupski & van Breugel 2024).
    
    trajstart and trajstop are in terms of 'time stamp', in ms, where 0 is
    aligned to Flash_bool, and sample rate = 10 ms/frame
    trajstart = -500
    trajstop = 1000
    windspeed = ambient windspeed in cm/s
    
    Returns: 
        A list of trajectory dataframes with new computed variables
    
    '''
    ## Import the data.
    data = pd.read_csv(csv)
    if 'ground speed' not in data:
        data['ground speed']=(data['xvel']**2+data['yvel']**2)**0.5
    
    ## Select time-segmented trajectories.
    data_2 = data[data['time stamp']>trajstart]
    data_3 = data_2[data_2['time stamp']<=trajstop]
    data_3.reset_index()
    
    n_traj = sum(data_3['Flash_bool'])
    startlocs = [0] + list(np.where(np.diff(data_3['time stamp'])<0)[0]+1)
    endlocs = list(np.where(np.diff(data_3['time stamp'])<0)[0]) + [len(data_3['time stamp'])]
    
    ## Compute variables of interest.
    traj_list = []
    new_index = pd.Index(np.linspace(trajstart+10.0,trajstop,num=int((trajstop-trajstart)/10)))
    for n in range(n_traj):
        traj = data_3.iloc[startlocs[n]:endlocs[n]+1,:].copy()
        traj['windspeed'] = np.ones_like(traj['time stamp'])*windspeed
        a = unwrap_angle(traj['heading'].values, 5)
        _, da = pynumdiff.finite_difference.first_order(a,0.01)
        traj['ang vel'] = da  # units per sec if you did the dt param correctly
        traj['abs ang vel'] = abs(da)
        traj['heading_unwrapped'] = a
        g = traj['ground speed'].copy()
        smoothg, dg = pynumdiff.linear_model.savgoldiff(g,10,[3,10,10])
        traj['ground speed smooth dot'] = dg.copy()
        o = wrapToPi(np.abs(traj['heading']))
        # traj['up-down-cross']=0
        # traj['up-down-cross'] = np.where( o<=np.pi/4, 1, 
        #                                  traj['up-down-cross'])
        # traj['up-down-cross'] = np.where( o>3*np.pi/4, -1, 
        #                                  traj['up-down-cross'])

        traj['up-down-cross']=0
        traj['up-down-cross'] = np.where( o<=np.pi/8, 1.0, 
                                          traj['up-down-cross'])
        traj['up-down-cross'] = np.where( (o>np.pi/8) & (o<=3*np.pi/8), 0.5, 
                                          traj['up-down-cross'])
        # traj['up-down-cross'] = np.where( o>3*np.pi/8 & o<=5*np.pi/8, 0.0,
        #                                  traj['up-down-cross'])
        traj['up-down-cross'] = np.where( (o>5*np.pi/8) & (o<=7*np.pi/8), -0.5,
                                          traj['up-down-cross'])
        traj['up-down-cross'] = np.where( o>7*np.pi/8, -1.0,
                                          traj['up-down-cross'])
        traj['up-down-cross'] = np.where( o==np.nan, np.nan,
                                          traj['up-down-cross'])
        traj['w_over_g'] = (traj['windspeed']/100) / g
        traj = traj.set_index('time stamp')
        traj = traj.reindex(new_index)
        traj_list.append(traj)
    
    return traj_list

def set_up_down_cross(traj,num=8,centered_on=0):
    '''
    Parameters
    ----------
    num : int, optional
        Number of upwind/downwind values to compute between -1 and 1 as a
        success metric for flies performing olfactory navigation. The default 
        is 8.
    centered_zero : float, optional
        Direction in radians for the center of upwind (will receive value of 
        1). The default is 0.0.

    Returns
    -------
    None.

    '''
    


def get_distribution(traj_list,variable,start,end,plotting,bins=30):
    '''
    Parameters
    ----------
    traj_list : list
        List of trajectory dataframes.
    variable : string
        Column label in traj_list dfs for desired data, e.g. 'heading'.
    start and end : int
        Start and end dataframe indices over which to compile data.
    plotting : bool
        Toggle plot of distribution.

    Returns
    -------
    Curated dataset used to plot histogram.

    '''
    out = np.empty((len(traj_list),traj_list[0][start:end].shape[0]))
    for i in range(len(traj_list)):
        out[i,:] = traj_list[i][variable][start:end]
        
    if plotting:
        if bins is None:
            bins=np.linspace(np.min(out),np.max(out),100)
        fig,ax = plt.subplots();
        ax.hist(np.reshape(out,out.shape[0]*out.shape[1]),bins);
        ax.set_title(variable + ', range: '+str(start)+':'+str(end));
        #ax.set_xlim(0,5);
    
    return out


def plot_log_probability(datalist,bins):
    '''
    This plots multiple PMF (probability mass function), not PDF, on a log-
    scale y-axis.
    
    Parameters
    ----------
    datalist : list of arrays
        Each list item is an output from get_distribution with certain 
        start/end indices. (e.g. -200:10, aka baseline data)
    bins : sequence
        Sets the bins (use np.linspace) for the histogram.

    Returns
    -------
    None.

    '''
    
    fig,ax = plt.subplots();
    for r in range(len(datalist)):
        data = datalist[r]
        d = np.histogram(data[np.isfinite(data)],bins=bins)
        ax.plot(d[1][1:],d[0]/np.sum(d[0]));

    
def mean_log_probability(data,bins):
    '''
    This will iterate over 2D trajectory data to compute the PMF for 
    each trajectory, and then the mean and SE of the set.

    Parameters
    ----------
    data : array
        Output from get_distribution. Rows are trajectories and columns are
        timepoints.
    bins : sequence
        Defines the bins for the histograms.

    Returns
    -------
    m : float array
        Mean of PMF of the trajectories.
    sem : float array
        SE of trajectory PMFs.

    '''
    
    n_traj = data.shape[0]
    pmfs = np.empty((n_traj,len(bins)-1))
    
    for traj in range(n_traj):
        pmfs[traj,:],_ = np.histogram(data[traj,np.isfinite(data[traj,:])],bins)
        pmfs[traj,:] = pmfs[traj,:]/np.nansum(pmfs[traj,:])
    
    m = np.nanmean(pmfs,axis=0)
    se = np.nanstd(pmfs,axis=0)/np.sqrt(n_traj)
    
    return m, se


def sep_by_value(traj_list,variable,start,end,threshold):
    '''
    This function will take a trajectory list and sort the trajectories into 
    two new lists based on the mean value of the specified variable and its 
    relationship to the threshold.

    Parameters
    ----------
    traj_list : list
        List of trajectory dataframes.
    variable : string
        Column label in traj_list dfs for desired data, e.g. 'heading'.
    start and end : int
        Start and end indices over which to compile data.
    threshold : float
        Value on which to sort the data 

    Returns
    -------
    below : list
        List of trajectory dataframes whose mean value of varibale during 
        the time period start->end is BELOW the threshold.
    above : list
        List of trajectory dataframes whose mean value of variable during 
        the time period start->end MEETS OR SURPASSES the threshold.

    '''
    below = []
    above = []
    for i in range(len(traj_list)):
        testdata = traj_list[i][variable].loc[start:end]
        # if np.nanmean(np.abs(testdata))<threshold:
        if np.nanmean(testdata)<threshold:
            below.append(traj_list[i])
        else: above.append(traj_list[i])
        
    return below, above

def sep_by_abs_value(traj_list,variable,start,end,threshold):
    '''
    This function will take a trajectory list and sort the trajectories into 
    two new lists based on the mean value of the specified variable and its 
    relationship to the threshold.

    Parameters
    ----------
    traj_list : list
        List of trajectory dataframes.
    variable : string
        Column label in traj_list dfs for desired data, e.g. 'heading'.
    start and end : int
        Start and end indices over which to compile data.
    threshold : float
        Value on which to sort the data 

    Returns
    -------
    below : list
        List of trajectory dataframes whose mean value of varibale during 
        the time period start->end is BELOW the threshold.
    above : list
        List of trajectory dataframes whose mean value of variable during 
        the time period start->end MEETS OR SURPASSES the threshold.

    '''
    below = []
    above = []
    for i in range(len(traj_list)):
        testdata = traj_list[i][variable].loc[start:end]
        if np.nanmean(np.abs(testdata))<threshold:
        # if np.nanmean(testdata)<threshold:
            below.append(traj_list[i])
        else: above.append(traj_list[i])
        
    return below, above

def compute_angular_dispersion(traj_list,start,end,plotting=False,ax=None):
    '''
    
    '''
    end+=10
    angdisps = np.empty((len(traj_list),int(np.ceil((end-start)/10))))
    for i in range(len(traj_list)):
        angdisps[i,:] = np.abs(traj_list[i]['heading_unwrapped'].loc[start:end]-traj_list[i]['heading_unwrapped'].loc[start])
        # angdisps[i,:] = traj_list[i]['heading_unwrapped'].loc[start:end]-traj_list[i]['heading_unwrapped'].loc[start]
    
    m = np.nanmean(angdisps,axis=0)
    se = np.nanstd(angdisps,axis=0)/np.sqrt(len(traj_list))
    
    if plotting:
        if ax is None:
            fig,ax = plt.subplots();
        ax.plot(np.linspace(start,end,int(np.ceil((end-start)/10))),np.nanmean(angdisps,axis=0));
        ax.fill_between(np.linspace(start,end,int(np.ceil((end-start)/10))),
                        m-se,m+se,alpha=0.5)
    
    return angdisps


def compute_airspeed_angle(v_para,v_perp,w,phi,zeta):
    '''
    To compute airspeed angle for David Stupski published data.

    Parameters
    ----------
    v_para : 1D numpy array
        Vector of forward velocities. Produced by decomposing groundspeed 
        vector along heading.
    v_perp : 1D numpy array
        Vector of lateral velocities. Produced by decomposing groundspeed
        vector along heading.
    w : 1D numpy array
        Vector of windspeeds.
    phi : 1D numpy array
        Vector of headings.
    zeta : 1D numpy array
        Vector of wind directions.

    Returns
    -------
    None.

    '''
    a_para = v_para - w * np.cos(phi - zeta)  # in fly ref frame
    a_perp = v_perp + w * np.sin(phi - zeta)  # in fly ref frame
    gamma = np.arctan2(a_perp, a_para)  # air velocity angle   ### in fly ref frame?
    
    return gamma



def _sample_cdf(alpha, resolution=100., axis=None):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251224.
    
    Helper function for circ_kuipertest.
    Evaluates CDF of sample in thetas.

    :param alpha: sample (in radians)
    :param resolution: resolution at which the cdf is evaluated (default 100)
    :param axis: axis along which the cdf is computed
    :returns: points at which cdf is evaluated, cdf values

    """

    if axis is None:
        alpha = alpha.ravel()
        axis = 0
    bins = np.linspace(0, 2 * np.pi, resolution + 1)
    old_shape = alpha.shape
    alpha = alpha % (2 * np.pi)

    alpha = alpha.reshape((alpha.shape[0], int(np.prod(alpha.shape[1:])))).T
    cdf = np.array([np.histogram(a, bins=bins)[0]
                    for a in alpha]).cumsum(axis=1) / float(alpha.shape[1])
    cdf = cdf.T.reshape((len(bins) - 1,) + old_shape[1:])

    return bins[:-1], cdf

def _kuiper_lookup(n, k):
    """
    CEM lifted this function, and 'kuiper_table.npy', from: 
        github.com/circstat/pycircstat
    on 20251224.
    """
    #ktable = load_kuiper_table()
    ktable = np.load('kuiper_table.npy')

    alpha = np.asarray([.10, .05, .02, .01, .005, .002, .001])
    nn = ktable[:, 0]

    isin = (nn == n)
    if np.any(isin):
        row = np.where(isin)[0]
    else:
        row = len(nn) - np.sum(n < nn) - 1

        if row == 0:
            raise ValueError('N too small.')
        else:
            warnings.warn(
                'N=%d not found in table, using closest N=%d present.' %
                (n, nn[row]))

    idx = (ktable[row, 1:] < k).squeeze()
    if np.any(idx):
        return alpha[idx].min()
    else:
        return 1.


def kuiper(alpha1, alpha2, res=100, axis=None):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251224.
    
    The Kuiper two-sample test tests whether the two samples differ
    significantly.The difference can be in any property, such as mean
    location and dispersion. It is a circular analogue of the
    Kolmogorov-Smirnov test.

    H0: The two distributions are identical.
    HA: The two distributions are different.

    :param alpha1: fist sample (in radians)
    :param alpha2: second sample (in radians)
    :param res:    resolution at which the cdf is evaluated (default 100)
    :returns: p-value and test statistic
              p-value is the smallest of .10, .05, .02, .01, .005, .002,
              .001, for which the test statistic is still higher
              than the respective critical value. this is due to
              the use of tabulated values. if p>.1, pval is set to 1.

    References: [Batschelet1980]_ p. 112

    """

    if axis is not None:
        assert alpha1.shape[
               1:] == alpha2.shape[
                      1:], "Shapes of alphas not consistent with computation along axis."
    n, m = alpha1.shape[axis], alpha2.shape[axis]

    _, cdf1 = _sample_cdf(alpha1, res, axis=axis)
    _, cdf2 = _sample_cdf(alpha2, res, axis=axis)
    
    # fig,ax=plt.subplots(figsize=(5,5))
    # ax.plot(cdf1)
    # ax.plot(cdf2)
    # plt.savefig('ann67trainvstest_cdf.svg')

    dplus = np.atleast_1d((cdf1 - cdf2).max(axis=axis))
    dplus[dplus < 0] = 0.
    dminus = np.atleast_1d((cdf2 - cdf1).max(axis=axis))
    dminus[dminus < 0] = 0.

    k = n * m * (dplus + dminus)
    effect = dplus + dminus
    mi = np.min([m, n])
    fac = np.sqrt(n * m * (n + m))
    pval = np.asarray([_kuiper_lookup(mi, kk / fac)
                       for kk in k.ravel()]).reshape(k.shape)
    return pval, k, effect


def omnibus(alpha, w=None, sz=np.radians(1), axis=None):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251226.
    
    Computes omnibus test for non-uniformity of circular data. The test is also
    known as Hodges-Ajne test.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    Alternative to the Rayleigh and Rao's test. Works well for unimodal,
    bimodal or multimodal data. If requirements of the Rayleigh test are
    met, the latter is more powerful.

    :param alpha: sample of angles in radian
    :param w:      number of incidences in case of binned angle data
    :param sz:    step size for evaluating distribution, default 1 rad
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return m:    minimum number of samples falling in one half of the circle

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    alpha = alpha % (2 * np.pi)
    n = np.sum(w, axis=axis)

    dg = np.arange(0, np.pi, np.radians(1))

    m1 = np.zeros((len(dg),) + alpha.shape[1:])
    m2 = np.zeros((len(dg),) + alpha.shape[1:])

    for i, dg_val in enumerate(dg):
        m1[i, ...] = np.sum(
            w * ((alpha > dg_val) & (alpha < np.pi + dg_val)), axis=axis)
        m2[i, ...] = n - m1[i, ...]

    m = np.concatenate((m1, m2), axis=0).min(axis=axis)

    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    A = np.empty_like(n)
    pval = np.empty_like(n)
    idx50 = (n > 50)

    if np.any(idx50):
        A[idx50] = np.pi * np.sqrt(n[idx50]) / 2 / (n[idx50] - 2 * m[idx50])
        pval[idx50] = np.sqrt(2 * np.pi) / A[idx50] * \
                      np.exp(-np.pi ** 2 / 8 / A[idx50] ** 2)

    if np.any(~idx50):
        pval[~idx50] = 2 ** (1 - n[~idx50]) * (n[~idx50] - \
                                               2 * m[~idx50]) * special.comb(n[~idx50], m[~idx50])

    return pval.squeeze(), m


def rayleigh(alpha, w=None, d=None, axis=None):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251226.
    
    Computes Rayleigh test for non-uniformity of circular data.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    # if axis is None:
    # axis = 0
    #     alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return pval, z

def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251226.
    
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r

def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    """
    CEM lifted this function from: 
        github.com/circstat/pycircstat
    on 20251226.
    
    """
    
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))
