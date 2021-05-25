#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:06:53 2020

@author: korepashki

"""

# settings:
global min_struct_el
min_struct_el = 7 
max_number_baseline_iterations = 16 # number of iterations in baseline search


import numpy as np
import skimage.morphology as skm
# from csaps import csaps
from itertools import groupby, count
# from scipy import stats
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend
import matplotlib as mpl
# from test_synth_BL import trig_bl
from scipy.optimize import least_squares

try:
    from csaps import csaps
except ImportError as e:
    print('Random baseline and morph/pspline algorithm require csaps module. You may install it by running \n pip install -U csaps')
    
try:
    import skued
except ImportError as e:
    print('Wavelet algorithm requires skued module. Depending on your installation you may install it by running \n pip install scikit-ued \n or \n conda config –add channels conda-forge conda install scikit-ued')


mpl.rcParams['figure.dpi'] = 300 # default resolution of the plot
import matplotlib.pyplot as plt
plt.style.use('bmh')

def spectrum_baseline (y, x=[], display=0, algorithm='derpsalsa', wl_level=9):
    """
    ----------
    x, y : spectral data
    display : 0, 1 or 2, optional
        The default is 2.
        0: no display, 1: final, 2: verbose with plots
    algorithm : derpsalsa, psalsa, als, morph_pspline, Koch, wavelet
        The default is 'derpsalsa'.

    Returns
    baseline
    """
    if len(x)==0:
        x = np.linspace(0, 1000, num=len(y))

    if display > 0:
        print('algorithm = ', algorithm, ' , starting')
    if algorithm == 'psalsa':
        baseline = psalsa_baseline(x, y, display)
    if algorithm == 'derpsalsa':
        baseline = derpsalsa_baseline(x, y, display)
    elif algorithm == 'als':
        baseline = baseline_als(x, y, display)
    elif algorithm == 'Koch':
        baseline = kolifier_morph_baseline(x, y, display)
    elif algorithm == 'morph_pspline':
        baseline = morph_pspline_baseline(x, y, display)
    elif algorithm == 'wavelet':
        from skued import baseline_dt, spectrum_colors
        max_iteration_number = 512
        baseline = baseline_dt(y, level = wl_level, max_iter = max_iteration_number, wavelet = 'qshift3')
        if display >= 1:
            levels = list(range(wl_level-2,wl_level+3))
            colors = spectrum_colors(levels)
            for l, c in zip(levels, colors):
                if l == wl_level:
                    continue
                bltmp = baseline_dt(y, level = l, max_iter = max_iteration_number, wavelet = 'qshift3')
                plt.plot(x, bltmp, color = c)
            plt.plot(x, y, 'r',
                     x, baseline, 'k--', linewidth = 2);
            plot_annotation = 'wavelet bl; black-- at level ' + str(wl_level)
            plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
            plt.show()
    return baseline



def baseline_als(x, y, display=2, als_lambda=5e6, als_p_weight=3e-6):
    """ asymmetric baseline correction
    Code by Rustam Guliev ~~ https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    parameters which can be manipulated:
    als_lambda  ~ 5e6
    als_p_weight ~ 3e-6
    (found from optimization with random smooth BL)
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(max_number_baseline_iterations):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: # 
    if display > 1:
        plt.plot(x, y - z, 'r', # subtracted spectrum
                  x, y, 'k',    # original spectrum
                  x, baseline, 'b');
        plot_annotation = 'ALS baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return baseline


def psalsa_baseline(x, y, display=2, als_lambda=6e7, als_p_weight=1.1e-3):
    """ asymmetric baseline correction
    Algorithm by Sergio Oller-Moreno et al.
    Parameters which can be manipulated:
    als_lambda  ~ 6e7
    als_p_weight ~ 1.1e-3
    (found from optimization with random 5-point BL)
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    # k = 10 * morphological_noise(y) # above this height the peaks are rejected
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * np.exp(-(y-z)/peakscreen_amplitude) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: # 
    if display > 1:
        plt.plot(x, y - z, 'r',
                  x, y, 'k',
                  x, baseline, 'b');
        plot_annotation = 'psalsa baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return baseline


def derpsalsa_baseline(x, y, display=2, als_lambda=5e7, als_p_weight=1.5e-3):
    """ asymmetric baseline correction
    Algorithm by Sergio Oller-Moreno et al.
    Parameters which can be manipulated:
    als_lambda  ~ 5e7
    als_p_weight ~ 1.5e-3
    (found from optimization with random 5-point BL)
    """

    # 0: smooth the spectrum 16 times
    #    with the element of 1/100 of the spectral length:
    zero_step_struct_el = int(2*np.round(len(y)/200) + 1)
    y_sm = molification_smoothing(y, zero_step_struct_el, 16)
    # compute the derivatives:
    y_sm_1d = np.gradient(y_sm)
    y_sm_2d = np.gradient(y_sm_1d)
    # weighting function for the 2nd der:
    y_sm_2d_decay = (np.mean(y_sm_2d**2))**0.5
    weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
    # weighting function for the 1st der:
    y_sm_1d_decay = (np.mean((y_sm_1d-np.mean(y_sm_1d))**2))**0.5
    weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)
    
    weifunc = weifunc1D*weifunc2D

    # exclude from screenenig the edges of the spectrum (optional)
    weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1

    # estimate the peak height
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8 # /8 is good, because this is a characteristic height of a tail
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    # k = 10 * morphological_noise(y) # above this height the peaks are rejected
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * weifunc * np.exp(-((y-z)/peakscreen_amplitude)**2/2) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: # 
    if display > 1:
        plt.plot(x, y - z, 'r',
                  x, y, 'k',
                  x, baseline, 'b');
        plot_annotation = 'derpsalsa baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return baseline



def kolifier_morph_baseline(x, y, display=2):
    """ Morphological baseline algorithm by Koch et al.: 10.1002/jrs.5010
    Structure element is calculated as in JJGV paper: 10.1002/jrs.5130
    
    """
    baseline = np.zeros_like(y)
    subtractedspectrum = y
    i = 0
    current_struct_el = find_structure_element_with_roll(subtractedspectrum)
    while True :
        baseline_tmp = skm.erosion (subtractedspectrum, np.ones(current_struct_el))
        baseline_tmp = molification_smoothing (baseline_tmp, current_struct_el, 1)
        baseline += baseline_tmp
        subtractedspectrum = y - baseline
        i += 1
        #@Test&Debug:
        # if display > 1:
        #     print('iteration number ', i, ', structure element is ', current_struct_el)
        #     plt.plot(x, subtractedspectrum, 'r', x, y, 'k', x, baseline, 'b');
        #     plot_annotation = '''Koch's 'erosion + molification, i = '''+str(i)
        #     plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        #     plt.show()
        #     tmpv = input("""press 'k Enter' to stop or 'Enter' to continue """)
        #     if tmpv == 'k':
        #         print('''okay, let's stop here''')
        #         break
        if i >= 5 :
            if display > 1:
                print('reached the maximum number of erosion-molifications, stopping here')
                plt.plot(x, y, 'k',
                         x, baseline, '--r')
            break
    return baseline


def morph_pspline_baseline(x, y, display=2, patch=1, smoothing_penalty_at_baseline=0.1):
    """
    Morphology-based cubic p-spline baseline
    Gonzales-Vidal et al. JRS 2017 DOI 10.1002/jrs.5130    
    If patch=1 add to the K2 knots the edges of the spectrum
     (otherwise you may get insane deviation from the expected baseline)
    """
    current_struct_el = find_structure_element(y)
    rmsnoise = morphological_noise(y)
    
    opening_modified = skm.opening(y, np.ones(current_struct_el))
    opening_modified = np.minimum(opening_modified,
                                  0.5*(skm.dilation(y, np.ones(current_struct_el)) +
                                       skm.erosion (y, np.ones(current_struct_el))))

    knots_K2 = np.where(np.abs(y - opening_modified) < rmsnoise*1e-11)[0]
    if patch > 0:
        # correction for end-points of the spectrum and K2 knots:
        #   if the 1st and the last points are not in K2,
        #   then add them with small weights
        if knots_K2[0] != 0:
            knots_K2 = np.insert(knots_K2, 0, 0)
        if knots_K2[-1] != len(x)-1:
            knots_K2 = np.append(knots_K2, len(x)-1)
            
    baseline = csaps(x[knots_K2], y[knots_K2], x, smooth=smoothing_penalty_at_baseline)
    subtractedspectrum = y - baseline
    if display > 1:
        plt.plot(x, subtractedspectrum, 'r',
                 x, y, 'k',
                 x, baseline, 'b');
        plot_annotation = 'opening + p-spline'
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()
    return baseline



def morphological_smoothing (wn, rawspectrum, smoothing_penalty_at_denoising=1) :
    """
    function that can be used for smoothing and/or de-noising
    JRS 2017, DOI 10.1002/jrs.5010
    smoothing_penalty_at_denoising=1 is for cubic spline (good choice for de-noising);
    for penalized spline you can set it, say, to 0.1
    """
    global min_struct_el
    rmsnoise = morphological_noise(rawspectrum)
    basic_m_smoothing = skm.opening(rawspectrum, np.ones(min_struct_el))
    basic_m_smoothing = skm.closing(basic_m_smoothing, np.ones(min_struct_el))
    knots_K1 = np.where(np.abs(basic_m_smoothing - rawspectrum) < rmsnoise)
    smoothed_spectrum = csaps(wn[knots_K1], rawspectrum[knots_K1], wn, smooth=smoothing_penalty_at_denoising)
    return smoothed_spectrum


def molification_smoothing (rawspectrum, struct_el, number_of_molifications):
    """ Molifier kernel here is defined as in the work of Koch et al.:
        JRS 2017, DOI 10.1002/jrs.5010
        The structure element is in pixels, not in cm-1!
        struct_el should be odd integer >= 3
    """
    molifier_kernel = np.linspace(-1, 1, num=struct_el)
    molifier_kernel[1:-1] = np.exp(-1/(1-molifier_kernel[1:-1]**2))
    molifier_kernel[0] = 0; molifier_kernel[-1] = 0
    molifier_kernel = molifier_kernel/np.sum(molifier_kernel)
    denominormtor = np.convolve(np.ones_like(rawspectrum), molifier_kernel, 'same')
    smoothline = rawspectrum
    i = 0
    for i in range (number_of_molifications) :
        smoothline = np.convolve(smoothline, molifier_kernel, 'same') / denominormtor
        i += 1
    return smoothline


def find_structure_element_with_roll(rawspectrum):
    """
    find_structure_element_with_roll is the iterative search of the structuring element
        by increasing it from min_struct_el_at_baseline until the opening stops changing.
    The 'roll' is used to take into account the edge effect.
    In the original approach of Gonzalez-Vidal no roll is used
        and the structure element may give unreasonably high values for the baseline with pronounced slope.
    """
    shift_width = round(len(rawspectrum)/2) # this is the roll spacing
    structure_element = np.minimum(find_structure_element(rawspectrum), 
        find_structure_element(np.roll(rawspectrum, shift_width)))
    #@Test&Debug: # print('structure element by roll is ', structure_element)
    if structure_element > len(rawspectrum)/3 + 2:
        structure_element = int(2*np.round(len(rawspectrum)/6) + 1)
        #@Test&Debug: # print('structure element should not exceed 1/3 of the spectrum length, limiting to ', structure_element)
    return structure_element


def find_structure_element(rawspectrum, x_sampling_scale=2, min_struct_el_at_baseline=7):
    """
    find_structure_element is the iterative search of the structuring element
        by increasing it from min_struct_el_at_baseline until the opening stops changing.
    This is the original algorithm by Gonzalez-Vidal https://doi.org/10.1002/jrs.5130
    x_sampling_scale controls an increment on iterative expanding of the structure element.
    For experimental systems where the x-step is small, it is useful to set x_sampling_scale to 2 or 4,
    otherwise it can be set to 1.
    min_struct_el_at_baseline is the starting point of the search, default is 7 pixels.
    """
    _count = 0
    current_struct_el = min_struct_el_at_baseline
    opening3array = np.zeros([len(rawspectrum), 3])
    rmsnoise = morphological_noise(rawspectrum)
    while True :
        opening_current = skm.opening(rawspectrum, np.ones(current_struct_el))
        opening3array = np.insert(opening3array, 0, [opening_current], axis=1 )
        opening3array = opening3array[:, 0:3]
        if _count < 3:
            _count += 1
            current_struct_el += 2*x_sampling_scale
            continue
        if np.linalg.norm(opening3array[:,1] - opening3array[:,0]) + np.linalg.norm(opening3array[:,2] - opening3array[:,1])  < rmsnoise*0.001 :
            current_struct_el -= 4*x_sampling_scale # structure element found!
            break
        current_struct_el += 2*x_sampling_scale
        _count += 1
    #@Test&Debug: #  print('opening converged, structure element = ', current_struct_el)
    return current_struct_el


def morphological_noise(rawspectrum):
    thenoise = rawspectrum
    thenoise = (thenoise - np.roll(thenoise, min_struct_el-2))**2
    # Remove the highest values that correspond to edge effect
    #  and, symmetrically, two smalles values
    for i in range (min_struct_el+1) :
        thenoise = np.delete(thenoise, np.argmax(thenoise))
        thenoise = np.delete(thenoise, np.argmin(thenoise))
        thenoise = np.delete(thenoise, np.argmin(thenoise))
    rmsnoise = (np.average(thenoise))**0.5
    #@Test&Debug: #  print ('morphological noise is ', rmsnoise)
    return rmsnoise


def random_baseline(x, number_of_knots=5, smoothing_penalty=0.1, display=0):
    # a: define the knots
    knots = np.rint(np.linspace(0, len(x)-1, num=number_of_knots)).astype(int)
    y_for_knots = 2*(np.random.random(number_of_knots) - 0.5)
    # b: compute spline
    random_baseline = csaps(x[knots], y_for_knots, x, smooth=smoothing_penalty)
    # c: #@Test&Debug:
    if display > 0:
        plt.plot(x, random_baseline, 'k--')
        plt.plot(x[knots], random_baseline[knots], 'o', mfc='none', ms = 6, mec='red')
        plt.show()
    return random_baseline


if __name__ == '__main__':
    synthetic_spectrum = np.genfromtxt('test_data_synthetic_spectrum.txt')
    available_algorithms = ('derpsalsa', 'psalsa', 'als', 'morph_pspline', 'Koch', 'wavelet')
    import random
    x = synthetic_spectrum[:,0]
    synth_y = synthetic_spectrum[:,1]
    synthetic_bl = 500*random_baseline(x)
    experimental_spectrum = np.genfromtxt('test_data_experimental_spectrum.txt')
    exp_y = experimental_spectrum[:,1]
    blp = spectrum_baseline(exp_y+synthetic_bl, display=0, algorithm='psalsa')
    bla = spectrum_baseline(exp_y+synthetic_bl, display=0, algorithm='als')
    bldp = spectrum_baseline(exp_y+synthetic_bl, display=0, algorithm='derpsalsa')
    plt.figure(figsize=(4.8, 2.8), constrained_layout=True)
    plt.plot(x, exp_y+synthetic_bl, 'k', label='exp.spec. + random smooth line');
    plt.plot(x, bldp, 'r', label='derpsalsa')
    plt.plot(x, blp, 'b', label='psalsa');
    plt.plot(x, bla, 'g', label='als');
    plt.legend(loc='upper left', frameon=False)
    plt.tick_params(labelleft=False)
    plt.xlim((0, 3800))
    plt.ylabel('Raman intensity')
    plt.xlabel("wavenumber / cm$^{-1}$")
    plt.show()
    print('you may run the script again to test a different random smooth line')
    print('''otherwise, consider running the following line:\n_ = spectrum_baseline(exp_y+synthetic_bl, display=2, algorithm=random.choice(available_algorithms))''')
    
    exp_x = experimental_spectrum[:,0]
    # exp_y = experimental_spectrum[:,1]
    blp = spectrum_baseline(exp_y, display=0, algorithm='psalsa')
    bla = spectrum_baseline(exp_y, display=0, algorithm='als')
    bldp = spectrum_baseline(exp_y, display=0, algorithm='derpsalsa')
    blkoch = spectrum_baseline(exp_y, display=0, algorithm='Koch')
    blmorph_pspline = spectrum_baseline(exp_y, display=0, algorithm='morph_pspline')
    blwavelet = spectrum_baseline(exp_y, display=0, algorithm='wavelet')
    np.savetxt('bl_psalsa.txt', np.transpose([exp_x, blp]))
    np.savetxt('bl_als.txt', np.transpose([exp_x, bla]))
    np.savetxt('bl_derpsalsa.txt', np.transpose([exp_x, bldp]))
    np.savetxt('bl_koch.txt', np.transpose([exp_x, blkoch]))
    np.savetxt('bl_morph_pspline.txt', np.transpose([exp_x, blmorph_pspline]))
    np.savetxt('bl_wavelet.txt', np.transpose([exp_x, blwavelet]))