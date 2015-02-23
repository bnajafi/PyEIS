# -*- coding: utf-8 -*-

"""
Documentation
"""

import os
import datetime
import shutil
import sys
import platform

import numpy as np
from scipy.optimize import fmin as nelder_mead
from scipy.stats import linregress

from matplotlib.backends.backend_pdf import PdfPages

import circuit_decomposition as cdp

from errors import ParameterNameError, ParameterNumberError, ParameterValueError

import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
rc['figure.figsize'] = (8,6)
rc['savefig.dpi'] = 300
rc['font.family'] = 'serif'
rc['font.size'] = 12
rc['font.serif'] = 'Times New Roman'
rc['font.sans-serif'] = 'Arial'
rc['font.monospace'] = 'Courier New'
rc['mathtext.default']='rm'
rc['mathtext.fontset'] = 'stix'
rc['xtick.labelsize']=16
rc['ytick.labelsize']=16
rc['axes.titlesize']=20
rc['axes.labelsize']=18
rc['figure.subplot.hspace']=0.5
rc['figure.subplot.wspace']=0.5
rc['legend.numpoints'] = 1      # the number of points in the legend line
rc['legend.fontsize'] = 16
rc['legend.markerscale'] = 1    # the relative size of legend markers vs. original
rc['lines.linewidth'] = 1
rc['lines.markeredgewidth'] = 1
rc['lines.markersize']=4
rc['axes.unicode_minus']=True
from matplotlib.ticker import AutoMinorLocator, AutoLocator, FuncFormatter



#CONSTANTS
ARCHITECTURE, OS = platform.architecture()
if ARCHITECTURE == '32bit':
    RESULT_FORMATTING = '%+.8e'
elif ARCHITECTURE == '64bit':
    RESULT_FORMATTING = '%+.16e'

SUMMARY_RESULT_FORMATTING = '%+.4e'

PRM_NAMES = ['Names', 'Values', 'LBounds', 'UBounds', 'Fixed', 'LogRandomize', 'LogScan', 'Sign']
PRM_FORMATS = [(np.str_, 32)] + 3*[np.float64] + [np.int32]*4
PRM_FORMATS_STR = ['%s'] + [RESULT_FORMATTING] + ['%+.2e']*2 + 4*['%d']
FIT_SETTING_EXT = 'FitSet'
PRM_INIT_EXT = 'PrmInit'
PRM_END_EXT = 'PrmEnd'
PRM_MIN_EXT = 'PrmMin'
DATA_MIN_EXT = 'DataMin'
DATA_END_EXT = 'DataEnd'
PRM_ALL_RUN_EXT = 'PrmAll'
DATA_FILE_EXTS = ['dot', 'data']
SUMMARY_END_EXT = 'SumEnd'
SUMMARY_MIN_EXT = 'SumMin'

DEFAULT_RANDOM_LOOPS = 200
DEFAULT_XTOL = 1e-8
DEFAULT_FTOL = 1e-8
DEFAULT_MAXITER_PER_PARAMETER = 200
DEFAULT_MAXFUN_PER_PARAMETER = 200

HEADER_MINIMIZATION_ELEMENTS = ['Nb of Run',\
              'Valid',\
              'np.log10(D)',\
              'LCC Module',\
              'LCC Phase',\
              'LCC Re',\
              'LCC Im',\
              'slope Module',\
              'slope Phase',\
              'slope Re',\
              'slope Im',\
              'intercept Module',\
              'intercept Phase',\
              'intercept Re',\
              'intercept Im']







def _get_header_footer_par_file(filepath):
    """
    
    Parse *.par file and get the number of lines for the header and the footer.
    
    Parameters
    ----------
    filepath: str
        Absolute filepath of the *.par file.
        
    Returns
    -------
    skip_header: int
        Number of lines to skip from the top
        
    skip_footer: int
        Number of lines to skip from the bottom
        
    nbpoints: int
        Number of points
    """
    
    skip_header_counter = 0
    skip_footer_counter = 0
    nblines = 0
    header_flag=True
    footer_flag = False

    with open(filepath,'r') as fobj:
            for line in fobj:
                nblines += 1
                if line.startswith('<Segment1>'):
                    header_flag = False
                    skip_header_counter +=1
                elif line.startswith('</Segment1>'):
                    footer_flag = True
                    skip_footer_counter +=1
                elif header_flag == True:
                    skip_header_counter +=1
                elif footer_flag == True:
                    skip_footer_counter +=1

    skip_header_counter += 3
    nbpoints = nblines - skip_footer_counter - skip_header_counter 
    
    return (skip_header_counter, skip_footer_counter, nbpoints)


def _get_exp_data(filepath):

    r"""

    Get the data array of data files according to their extension.

    Supported files are .z files recorded by ZView software, .par files recorded by VersaStudio and .data files
    were the first three columns represent :math:`f`, :math:`\left ReZ`, :math:`ImZ`.
    
    Parameters
    -----------
    filepath: string
        Path to the data file.

    Returns
    --------
    data_array: 2d array
        Experimental data.
        
    """
    
    name, ext = os.path.basename(os.path.abspath(filepath)).split('.')
    extensions = ['z', 'par', 'data']
    if ext.lower() in extensions:
        
        if ext == 'par':
            skip_header, skip_footer, nbpoints = _get_header_footer_par_file(filepath)
            names = ['Segment',\
                     'Points',\
                     'E','I','t',\
                     'ADC Sync Input(V)',\
                     'I Range', 'Status', 'E Applied(V)',\
                     'f', 'E Real', 'E Imag', 'I Real', 'I Imag','ReZ', 'ImZ']
            usecols = (9, 14, 15)
            delimiter = ','
            converters=None

        elif ext.lower() == 'data':
            skip_header, skip_footer = 0,0
            usecols = (0,1,2)
            delimiter = '\t'
            converters = None

        elif ext == 'z':
            skip_header, skip_footer, nbpoints = 11,0,0
            usecols = (0, 4, 5)
            delimiter=','
            converters=None
        
        w, ReZ, ImZ = np.genfromtxt(fname=filepath, dtype=np.float64, comments='#', delimiter=delimiter,\
                  skiprows=0, skip_header=skip_header, skip_footer=skip_footer,\
                  converters=converters, missing='', missing_values=None, filling_values=None,\
                  usecols=usecols, names=None, excludelist=None,\
                  deletechars=None, replace_space='_', autostrip=False, case_sensitive=True,\
                  defaultfmt='f%i', unpack=True, usemask=False, loose=True, invalid_raise=True)
        
        return w, ReZ, ImZ
    else:
        raise Exception('FileType Error: File was not recognized')

    

def _import_prm_file(filepath):
    r""" DocString"""

    npnames = PRM_NAMES
    npformats = PRM_FORMATS
    dtypes = np.dtype({'names':npnames, 'formats':npformats})

    filepath = os.path.abspath(filepath)
    prm_array = np.loadtxt(filepath,\
                           comments='#',\
                           delimiter='\t',
                           dtype=dtypes,\
                           unpack=False)

    return prm_array


def _check_parameter_names(prm_array, symbolic_immittance):
    r""" DocString"""

    parameters, parameters_names = cdp._get_parameters(symbolic_immittance)

    nb_parameters = len(parameters)
    nb_parameters_input = len(prm_array['Names'])

    if nb_parameters != nb_parameters_input:
        message = 'The number of given parameters is not equal to the number of parameters in the expression of the immittance.'
        raise ParameterNumberError(message)

    for name in parameters_names:
        if name not in prm_array['Names']:
            message = 'Imported parameter names do not correspond to the parameter in the expression of the immittance.'
            raise ParameterNameError(message)




def _check_parameter_values(prm_array):

    mask, = np.where(prm_array['Values'] == 0)
    if mask.size > 0:
        message = 'Parameter values must be different from 0.'
        raise ParameterValueError(message)

    mask, = np.where(prm_array['LBounds'] == 0.0)
    if mask.size > 0:
        message = 'Lower bounds must be different from 0.'
        raise ParameterValueError(message)

    mask, = np.where(prm_array['UBounds'] == 0.0)
    if mask.size > 0:
        message = 'Upper bounds must be different from 0.'
        raise ParameterValueError(message)
    
    mask, = np.where(prm_array['LBounds'] >= prm_array['UBounds'])
    if mask.size > 0:
        message = 'Lower bounds cannot be higher than upper bounds.'
        raise ParameterValueError(message)

    mask, = np.where( (prm_array['Fixed'] != 1) & (prm_array['Fixed'] != 0) )
    if mask.size > 0:
        message = 'Fixed option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)

    mask, = np.where( (prm_array['LogRandomize'] != 1) & (prm_array['LogRandomize'] != 0) )
    if mask.size > 0:
        message = 'LogRandomize option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)

    mask, = np.where( (prm_array['LogScan'] != 1) & (prm_array['LogScan'] != 0) )
    if mask.size > 0:
        message = 'LogScan option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)


    mask, = np.where( (prm_array['Sign'] != 1) & (prm_array['Sign'] != 0) & (prm_array['Sign'] != -1) )
    if mask.size > 0:
        message = 'Sign option can only be positive, negaive or both i.e. +1, 0 or -1.'
        raise ParameterValueError(message)

    

def _get_mask_to_fit(prm_array):

    mask, = np.where(prm_array['Fixed'] == 0)
    return mask


def _get_mask_not_valid(prm_array):
    r""" DocString"""

    mask_to_fit = _get_mask_to_fit(prm_array)
    
    Values = prm_array['Values'][mask_to_fit]
    LBounds = prm_array['LBounds'][mask_to_fit]
    UBounds = prm_array['UBounds'][mask_to_fit]
    
    mask, = np.where( (Values > UBounds) | (Values < LBounds) )

    return mask


def _get_mask_logscan(prm_array):
    r""" DocString"""
    
    mask_to_fit = _get_mask_to_fit(prm_array)
    mask, = np.where(prm_array['LogScan'][mask_to_fit] == 1)

    return mask


def _check_validity_prm(prm_array):
    r""" DocString"""

    flag = False

    mask_not_valid = _get_mask_not_valid(prm_array)

    if mask_not_valid.size == 0:
        flag = True

    return flag


def _initialize_prm_from_immittance(symbolic_immittance):

    parameters, parameter_names = cdp._get_parameters(symbolic_immittance)
    dtypes = np.dtype({'names':PRM_NAMES, 'formats':PRM_FORMATS})
    prm_array = np.zeros(shape=(len(parameters),), dtype = dtypes)
    prm_array['Names'] = parameter_names

    return prm_array


def _update_prm(from_prm, to_prm):
    
    
    for ind, name in enumerate(to_prm['Names']):
        mask, = np.where(from_prm['Names'] == name)
        to_prm[ind] = from_prm[mask]

    return to_prm


def _get_random_prm_values(prm_array, all_parameters=False):
    r""" DocString"""

    mask_to_fit = _get_mask_to_fit(prm_array)

    #if mask_not_valid is an empty array
    #no change to prm_array
    #initial randomizing
    if all_parameters == False:
        mask_not_valid = _get_mask_not_valid(prm_array)
    else:
        mask_not_valid = np.arange(mask_to_fit.size)
        
    #2wo sub 1d arrays 'Values_to_fit' and 'Values' from prm_array field 'Values' are
    #necesssary for updating values after calculations
    #LBound, UBounds, LBounds and LogScan do not need subarrays -> only accessing values
    Values_to_fit = prm_array['Values'][mask_to_fit]
    Values = Values_to_fit[mask_not_valid]
    LBounds = prm_array['LBounds'][mask_to_fit][mask_not_valid]
    UBounds = prm_array['UBounds'][mask_to_fit][mask_not_valid]
    LogRandomize= prm_array['LogRandomize'][mask_to_fit][mask_not_valid]
    Sign = prm_array['Sign'][mask_to_fit][mask_not_valid]

    mask_linear, = np.where( LogRandomize == 0 )
    mask_log, = np.where( LogRandomize == 1 )
    mask_positive = np.where(Sign == 1)
    mask_negative = np.where(Sign == -1)
    mask_both = np.where(Sign == 0)
    random_sign = np.random.randint(-1,1,len(mask_both))
    random_sign[random_sign==0] = 1.0

    #in linear scale the random values are classically calculated by: value = low + random(0,1)*(up-low)
    #in log scale the random values are calculated using the logarihmique values of the limits: value = 10**( log10(low) + random(0,1)*(log10(up)-log(low)) )
    Values[mask_linear] = LBounds[mask_linear] + np.random.rand(mask_linear.size) * ( UBounds[mask_linear] - LBounds[mask_linear] )
    Values[mask_log] = 10**( np.log10(LBounds[mask_log]) + np.random.rand(mask_log.size) * ( np.log10(UBounds[mask_log]) - np.log10(LBounds[mask_log]) ) )

    Values[mask_positive] = Values[mask_positive]*1.0
    Values[mask_negative] = Values[mask_negative]*-1.0
    Values[mask_both] = Values[mask_both]*random_sign

    Values_to_fit[mask_not_valid] = Values
    prm_array['Values'][mask_to_fit] = Values_to_fit

    return prm_array


def _get_distance(I_exp, I_calc):
    r"""

    Compute the distance :math:`S` between :math:`I_{exp}` and
    :math:`I_{calc}`.
    The distance is computed by multiplying the distances on real and imaginary
    parts of :math:`Iph`:

    .. math::
        
        \Delta Re & = Re \, I_{exp} - Re \, I_{calc} \\
        \Delta Im & = Im \, I_{exp} - Im \, I_{calc} \\
        S = \sum weights*(\Delta Re^2 + \Delta Im^2)

    Parameters
    ----------
    I_exp: 1d numpy array
            Contains the complex values of the :math:`I_{exp}`.

    I_calc: 1d numpy array
            Contains the complex values of the :math:`I_{calc}`.

    Returns
    -------
    D: float
            The computed distance :math:`S` on real and imaginary parts of :math:`I`:.

    """
    mod_I_exp = np.absolute(I_exp)
    Re_I_exp = np.real(I_exp)
    Im_I_exp = np.imag(I_exp)

    Re_I_calc = np.real(I_calc)
    Im_I_calc = np.imag(I_calc)
    
    delta_Re = (Re_I_exp - Re_I_calc)
    delta_Im = (Im_I_exp - Im_I_calc)

    weights = 1.0/mod_I_exp**2

    S = np.sum(weights*(delta_Re**2 + delta_Im**2))
    
    return S
        


def _target_function(p, w, prm_array, I_exp, I_num):
    r"""DocString"""

    mask_to_fit = _get_mask_to_fit(prm_array)
    mask_logscan = _get_mask_logscan(prm_array)

    p0 = prm_array['Values'][mask_to_fit]
    p0[:] = p[:]
    p0[mask_logscan] = 10**p0[mask_logscan]
    prm_array['Values'][mask_to_fit] = p0

    I_calc = I_num(prm_array['Values'], w)
    
    S = _get_distance(I_exp, I_calc)

    return S


def _minimize(w, I_exp_complex, I_numeric, prm_array,\
             maxiter=None, maxfun=None, xtol=DEFAULT_XTOL, ftol=DEFAULT_FTOL,\
             full_output=True, retall=False, disp=False, callback=None):

    r"""DocString"""

    mask_to_fit = _get_mask_to_fit(prm_array)
    mask_logscan = _get_mask_logscan(prm_array)
    p0 = prm_array['Values'][mask_to_fit]
    p0[mask_logscan] = np.log10(p0[mask_logscan])

    popt ,fopt ,iteration ,funcalls ,warnflag = nelder_mead(_target_function,\
                                                              p0 ,\
                                                              args=(w,\
                                                                    prm_array,\
                                                                    I_exp_complex,\
                                                                    I_numeric),\
                                                              maxiter = maxiter,\
                                                              maxfun = maxfun,\
                                                              xtol = xtol,\
                                                              ftol = ftol,\
                                                              full_output=full_output,\
                                                              retall=retall,\
                                                              disp=disp,\
                                                              callback=callback)

    p0 = prm_array['Values'][mask_to_fit]
    p0[:] = popt[:]
    p0[mask_logscan] = 10**p0[mask_logscan]
    prm_array['Values'][mask_to_fit] = p0

    return prm_array, fopt


def _get_complex_parameters(z, deg=True):

    mod = np.absolute(z)
    phase = np.angle(z, deg=deg)
    Re = np.real(z)
    Im = np.imag(z)

    return mod, phase, Re, Im


def _get_LCC(I_exp_complex, I_calc_complex):
    r"""


    """

    mod_exp, phase_exp, Re_exp, Im_exp = _get_complex_parameters(I_exp_complex, deg=True)
    mod_calc, phase_calc, Re_calc, Im_calc = _get_complex_parameters(I_calc_complex, deg=True)

    slope_mod, intercept_mod, r_mod, p_mod, std_mod = linregress(mod_exp, mod_calc)
    slope_phase, intercept_phase, r_phase, p_phase, std_phase = linregress(phase_exp, phase_calc)
    slope_Re, intercept_Re, r_Re, p_Re, std_Re = linregress(Re_exp, Re_calc)
    slope_Im, intercept_Im, r_Im, p_Im, std_Im = linregress(Im_exp, Im_calc)

    return r_mod, r_phase, r_Re, r_Im,\
            slope_mod, slope_phase, slope_Re, slope_Im,\
            intercept_mod, intercept_phase, intercept_Re, intercept_Im



def _get_results_array(f, I_exp_complex, I_calc_complex):

    r"""

    Build the data array of the experimental and calculated data: :math:`f`,
    :math:`ReZ_{exp}`, :math:`ImZ_{exp}`, :math:`ReZ_{calc}` and :math:`ImZ_{calc}`

            

    
    Parameters
    ----------
    f: 1d numpy array
        Contains the frequency vector.

    I_exp_complex: 1d array
            Contains the complex values of :math:`I_{exp}`.

    I_calc_complex: 1d array
            Contains the complex values of :math:`I_{calc}`.

    Returns
    -------
    data_array: 2d array
            Array containing the .

    """
    header = '\t'.join([r'f /Hz',\
                        r'|I_exp| /A', r'Phase_exp /deg', r'Re I_exp /A' , r'Im I_exp /A',\
                        r'|I_calc| /A', r'Phase_calc /deg', r'Re I_calc /A' , r'Im I_calc /A',\
                        r'Res |I| /A', r'Res Phase /deg', r'Res Re I /A', r'Res Im I /A'])
    
    mod_exp, phase_exp, Re_exp, Im_exp = _get_complex_parameters(I_exp_complex, deg=True)
    mod_calc, phase_calc, Re_calc, Im_calc = _get_complex_parameters(I_calc_complex, deg=True)

    Res_mod = mod_exp - mod_calc
    Res_phase = phase_exp - phase_calc
    Res_Re = Re_exp - Re_calc
    Res_Im = Im_exp - Im_calc
    
    data_array = np.transpose(np.vstack((f,\
                                         mod_exp, phase_exp, Re_exp, Im_exp,\
                                         mod_calc, phase_calc, Re_calc, Im_calc,\
                                         Res_mod, Res_phase, Res_Re, Res_Im)))
    
    return (header, data_array)


def _save_results(run, process_id, fit_folder, datafilepath, circuit_str, f, mask, I_exp_complex, I_numeric,\
                  prm_user, prm_min_run, prm_end_run, distance_min_run, distance_end_run,\
                  minimization_results, header_minimization_results):

    name, ext = os.path.basename(datafilepath).split('.')
    N = prm_end_run['Names'].size
    w = 2*np.pi*f

    #save minimization results
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}.{5:s}'.format(fit_folder, name, circuit_str, process_id, run+1, PRM_ALL_RUN_EXT)
    np.savetxt(filepath, minimization_results,\
               fmt=['%d','%d','%+.4f'] + [RESULT_FORMATTING]*(len(HEADER_MINIMIZATION_ELEMENTS)-3 + N),\
               delimiter='\t',\
               newline='\n',\
               header=header_minimization_results)

    #save minimum
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run+1, np.log10(distance_min_run), PRM_MIN_EXT)
    np.savetxt(filepath, _update_prm(prm_min_run, prm_user), fmt=PRM_FORMATS_STR ,delimiter='\t' ,newline='\n', header='\t'.join(PRM_NAMES))

    I_calc_complex = I_numeric(prm_min_run['Values'], w)
    header, data = _get_results_array(w[mask], I_exp_complex[mask], I_calc_complex[mask])
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run+1, np.log10(distance_min_run), DATA_MIN_EXT)
    np.savetxt(filepath, data, fmt=RESULT_FORMATTING ,delimiter='\t' ,newline='\n' ,header=header)

    ext='pdf'
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-{5:s}-d{6:.4f}.{7:s}'.format(fit_folder, name, circuit_str, process_id, run+1, 'Min', np.log10(distance_min_run), ext)
    _save_pdf(filepath,\
             f, I_exp_complex, I_calc_complex,\
             mask, minimization_results, data)

    #save end
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run+1, np.log10(distance_end_run), PRM_END_EXT)
    np.savetxt(filepath, _update_prm(prm_end_run, prm_user), fmt=PRM_FORMATS_STR ,delimiter='\t' ,newline='\n', header='\t'.join(PRM_NAMES))

    I_calc_complex = I_numeric(prm_end_run['Values'], w)
    header, data = _get_results_array(w[mask], I_exp_complex[mask], I_calc_complex[mask])
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run+1, np.log10(distance_end_run), DATA_END_EXT)
    np.savetxt(filepath, data, fmt=RESULT_FORMATTING ,delimiter='\t' ,newline='\n' ,header=header)

    ext='pdf'
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-{5:s}-d{6:.4f}.{7:s}'.format(fit_folder, name, circuit_str, process_id, run+1, 'End', np.log10(distance_end_run), ext)
    _save_pdf(filepath,\
             f, I_exp_complex, I_calc_complex,\
             mask, minimization_results, data)


def _save_pdf(filepath,\
              f, I_exp_complex, I_calc_complex,\
              mask, minimization_results, data):

    pdf = PdfPages(filepath)
    scilimits = (1e-6,1e6)

    mod_exp, phase_exp, Re_exp, Im_exp = _get_complex_parameters(I_exp_complex, deg=True)
    mod_calc, phase_calc, Re_calc, Im_calc = _get_complex_parameters(I_calc_complex, deg=True)

    #Nyqusit Plot - fitting range
    plt.figure(figsize=(8,6))
    plt.title(r'ReZ vs ImZ (fitting range)')
    plt.xlabel(r'ReZ /$\Omega$')
    plt.ylabel(r'ImZ /$\Omega$')
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(Re_exp[mask], Im_exp[mask],'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(Re_calc[mask], Im_calc[mask],'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    #Nyqusit Plot - full range
    plt.figure(figsize=(8,6))
    plt.title(r'ReZ vs ImZ (full range)')
    plt.xlabel(r'ReZ /$\Omega$')
    plt.ylabel(r'ImZ /$\Omega$')
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(Re_exp, Im_exp,'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(Re_calc, Im_calc,'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    #Phase Plot - fitting range
    plt.figure(figsize=(8,6))
    plt.title(r'$\theta$ vs f (fitting range)')
    plt.xlabel(r'f /Hz')
    plt.ylabel(r'$\theta$ /$^{\circ}$')
    plt.xscale('log')
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(f[mask], phase_exp[mask],'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f[mask], phase_calc[mask],'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    #Phase Plot - full range
    plt.figure(figsize=(8,6))
    plt.title(r'$\theta$ vs f (full range)')
    plt.xlabel(r'f /Hz')
    plt.ylabel(r'$\theta$ /$^{\circ}$')
    plt.xscale('log')
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(f, phase_exp,'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f, phase_calc,'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()
    

    #Module Plot - fitting range
    plt.figure(figsize=(8,6))
    plt.title(r'|Z| vs f (fitting range)')
    plt.xlabel(r'f /Hz')
    plt.ylabel('|Z| /$\Omega$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(f[mask], mod_exp[mask],'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f[mask], mod_calc[mask],'r.-', markersize=4, linewidth=1, label='fit')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    #Module Plot - full range
    plt.figure(figsize=(8,6))
    plt.title(r'|Z| vs f (full range)')
    plt.xlabel(r'f /Hz')
    plt.ylabel('|Z| /$\Omega$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(f, mod_exp,'k-o' , markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f, mod_calc,'r.-', markersize=4, linewidth=1, label='fit')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()


    plt.figure(figsize=(8,6))
    mask_, = np.where(minimization_results[:,0] != 0)
    mask_valid, = np.where(minimization_results[mask_,1] == 1)
    plt.plot(minimization_results[mask_,0], minimization_results[mask_,2], color='k', marker='o', linestyle='-', linewidth=1, mfc='w', mec='k', markeredgewidth=1, label='Not Valid')
    plt.plot(minimization_results[mask_valid,0], minimization_results[mask_valid,2], color='g', marker='o', linestyle='None', linewidth=1, mfc='w', mec='g', markeredgewidth=1, label='Valid')
    plt.title('log10(D) vs no fit')
    plt.xlabel('No of minimization')
    plt.ylabel('log(D)')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    #Residuals Re
    plt.figure(figsize=(8,6))
    plt.title(r'Residuals Re')
    plt.xlabel(r'$ReZ_{calc} - ReZ_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:,11], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='b')
    pdf.savefig()
    plt.close()
    
    #Residuals Im
    plt.figure(figsize=(8,6))
    plt.title(r'Residuals Im')
    plt.xlabel(r'$ImZ_{calc} - ImZ_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:,12], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='b')
    pdf.savefig()
    plt.close()
    
    #Residuals Phase
    plt.figure(figsize=(8,6))
    plt.title(r'Residuals Phase')
    plt.xlabel(r'$\theta_{calc} - \theta_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:,10], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='b')
    pdf.savefig()
    plt.close()

    #Residuals Module
    plt.figure(figsize=(8,6))
    plt.title(r'Residuals Module')
    plt.xlabel(r'$|Z|_{calc} - |Z|_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:,9], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='b')
    pdf.savefig()
    plt.close()

    pdf.close()


def _get_summary(fit_folder, symbolic_immittance, numeric_immittance):

    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)
    PRM_end = []
    PRM_min = []
    fitsettings_filepath = ''
    prm_array = _initialize_prm_from_immittance(symbolic_immittance)


    run = 0
    for i in listfiles:
        ext = i.split('.')[-1]
        if ext == PRM_END_EXT:
            run+=1
            PRM_end.append(os.path.abspath(dirpath + '/' + i))
        elif ext == PRM_MIN_EXT:
            PRM_min.append(os.path.abspath(dirpath + '/' + i))
        elif ext in DATA_FILE_EXTS:
            datafilepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    fitsettings_fobj = open(fitsettings_filepath,'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n','')
        
    f_start, f_end = fitsettings_dict['Frequency Range (Hz)'].split(',')
    f_start, f_end = float(f_start), float(f_end)

    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    prm_user_filepath = os.path.abspath(fitsettings_dict['Parameter File'])
    circuit = fitsettings_dict['Circuit']
    circuit_str = circuit.replace('+','_s_').replace('/','_p_')

    f, ReZ, ImZ = _get_exp_data(datafilepath)
    w = 2*np.pi*f
    I_exp_complex = ReZ+1j*ImZ
    mask, = np.where( (f>=f_start) & (f<=f_end) )

    prm_user = _import_prm_file(prm_user_filepath)
    N = prm_user.size
    header = '\t'.join(HEADER_MINIMIZATION_ELEMENTS) + '\t'
    header += '\t'.join(prm_user['Names'])
    col = len(HEADER_MINIMIZATION_ELEMENTS) + prm_user['Names'].size

    summary_end = np.zeros(shape = (run, col), dtype=np.float64)
    for ind, i in enumerate(PRM_end):

        prm = _import_prm_file(os.path.abspath(i))
        prm_array = _update_prm(prm, prm_array)
        valid = _check_validity_prm(prm_array)

        I_calc_complex = numeric_immittance(prm_array['Values'], w)
        distance = _get_distance(I_exp_complex[mask], I_calc_complex[mask])
        LCC_results = _get_LCC(I_exp_complex[mask], I_calc_complex[mask])

        summary_end[ind,:] = np.hstack((ind+1, valid, np.log10(distance), LCC_results, prm['Values']))

    #sort over the log10(D)
    mask_end = np.argsort(summary_end[:,2])
    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '.' + SUMMARY_END_EXT)
    np.savetxt(filepath, X=summary_end[mask_end], fmt=['%d','%d','%+.4f'] + [RESULT_FORMATTING]*(len(HEADER_MINIMIZATION_ELEMENTS)-3 + N), delimiter='\t', header=header)
    
    
    summary_min = np.zeros(shape = (run, col), dtype=np.float64)
    for ind, i in enumerate(PRM_min):

        prm = _import_prm_file(os.path.abspath(i))
        prm_array = _update_prm(prm, prm_array)
        valid = _check_validity_prm(prm_array)

        I_calc_complex = numeric_immittance(prm_array['Values'], w)
        distance = _get_distance(I_exp_complex[mask], I_calc_complex[mask])
        LCC_results = _get_LCC(I_exp_complex[mask], I_calc_complex[mask])

        summary_min[ind,:] = np.hstack((ind+1, valid, np.log10(distance), LCC_results, prm['Values']))

    #sort over the log10(D)
    mask_end = np.argsort(summary_min[:,2])
    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '.' + SUMMARY_MIN_EXT)
    np.savetxt(filepath, X=summary_min[mask_end], fmt=['%d','%d','%+.4f'] + [RESULT_FORMATTING]*(len(HEADER_MINIMIZATION_ELEMENTS)-3 + N), delimiter='\t', header=header)

    

def _plot_summary(fit_folder):

    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)
    description_filepath = ''

    summary_end_filepath = ''
    summary_min_filepath = ''

    for i in listfiles:
        cuts = i.split('.')
        ext = cuts[-1]
        if ext == SUMMARY_END_EXT:
            summary_end_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == SUMMARY_MIN_EXT:
            summary_min_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    fitsettings_fobj = open(fitsettings_filepath,'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n','')

    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    prm_user_filepath = os.path.abspath(fitsettings_dict['Parameter File'])
    circuit = fitsettings_dict['Circuit']
    circuit_str = circuit.replace('+','_s_').replace('/','_p_')

    prm_user = _import_prm_file(prm_user_filepath)
    N = prm_user.size

    with open(summary_end_filepath,'r') as f:
        header_end = f.readline().replace('#','').replace('\n','').split('\t')
    
    summary_end = np.loadtxt(summary_end_filepath, comments='#',\
                             delimiter='\t',\
                             skiprows=0,\
                             ndmin=2,\
                             unpack=False)
    mask_end = np.argsort(summary_end[:,0])

    summary_min = np.loadtxt(summary_min_filepath, comments='#',\
                             delimiter='\t',\
                             skiprows=0,\
                             ndmin=2,\
                             unpack=False)
    mask_min = np.argsort(summary_min[:,0])

    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '-' + '0' + '-' + 'End' + '.' + 'pdf')
    pdf_end = PdfPages(filepath)
    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '-' + '0' + '-' + 'Min' + '.' + 'pdf')
    pdf_min = PdfPages(filepath) 

    colors = ['b', 'r', 'g', 'y', 'm', 'c', 'darkblue', 'darkred', 'darkgreen', 'lightblue', 'orange', 'lightgreen','k', 'gray']
    scilimits = (-4,4)


    params = summary_end[mask_end, len(HEADER_MINIMIZATION_ELEMENTS):]
    no_run = summary_end[mask_end, 0]
    for ind, name in enumerate(prm_user['Names']):
        if name[0] in ['R', 'D', 'M']:
            unit = '/$\Omega$'
        elif name[0] in['W']:
            unit = '/$\Omega \cdot s^{-1/2}$'
        elif name[0] in ['C']:
            unit = '/F'
        elif name[0] in ['L']:
            unit = '/H'
        elif name[0] in ['Q']:
            unit = '/$\Omega ^{-1} \cdot s^{n}$'
        elif name[0] in ['n','N']:
            unit = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.ticklabel_format(scilimits=scilimits)
        ax.grid()
        ax.set_title('{0:s} vs Run'.format(name))
        ax.set_xlabel('No Run')
        ax.set_ylabel(r'Values {1:s}'.format(name, unit))
        ax.plot(no_run, params[:,ind], color='k', marker='o', mfc='w', mec='k', ls='-', lw=1, ms=4, mew=1 )
        pdf_end.savefig(fig)
        plt.close(fig)


    params = summary_min[mask_min, len(HEADER_MINIMIZATION_ELEMENTS):]
    no_run = summary_min[mask_min, 0]
    for ind, name in enumerate(prm_user['Names']):
        if name[0] in ['R', 'D', 'M']:
            unit = '/$\Omega$'
        elif name[0] in['W']:
            unit = '/$\Omega \cdot s^{-1/2}$'
        elif name[0] in ['C']:
            unit = '/F'
        elif name[0] in ['L']:
            unit = '/H'
        elif name[0] in ['Q']:
            unit = '/$\Omega ^{-1} \cdot s^{n}$'
        elif name[0] in ['n','N']:
            unit = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.ticklabel_format(scilimits=scilimits)
        ax.grid()
        ax.set_title('{0:s} vs Run'.format(name))
        ax.set_xlabel('No Run')
        ax.set_ylabel(r'Values {1:s}'.format(name, unit))
        ax.plot(no_run, params[:,ind], color='k', marker='o', mfc='w', mec='k', ls='-', lw=1, ms=4, mew=1 )
        pdf_min.savefig(fig)
        plt.close(fig)
    

    pdf_end.close()
    pdf_min.close()

def _random_scan(w, prm_array, prm_user, I_exp_complex, symbolic_immittance, numeric_immittance, run, nb_run, loops=1, callback=None, args=None):

    prm_array_random = _initialize_prm_from_immittance(symbolic_immittance)
    prm_array_random_min = _initialize_prm_from_immittance(symbolic_immittance)

    #1st random scan
    prm_array_random = _get_random_prm_values(prm_array, all_parameters=True)
    I_calc_complex = numeric_immittance(prm_array_random['Values'], w)

    prm_array_random_min[:] = prm_array_random[:]
    distance = _get_distance(I_exp_complex, I_calc_complex)
    valid = _check_validity_prm(prm_array)

    distance_min = distance

    if callback is not None:
        callback(run, nb_run, 1, loops, distance, valid, prm_array, prm_user)

    for i in range(loops-1):
        prm_array_random = _get_random_prm_values(prm_array, all_parameters=True)
        I_calc_complex = numeric_immittance(prm_array_random['Values'], w)
        distance = _get_distance(I_exp_complex, I_calc_complex)
        valid = _check_validity_prm(prm_array)

        if callback is not None:
            callback(run, nb_run, i+2, loops, distance, valid, prm_array, prm_user)


        if valid == True:
            if distance < distance_min:
                distance_min = distance
                prm_array_random_min[:] = prm_array_random[:]
    

    return prm_array_random_min

def _callback_fit(run, nb_run, fit, nb_fit,\
              distance, valid,\
              LCC_results, prm_array, prm_user, additional_messages=[]):

    os.system('cls' if os.name == 'nt' else 'clear')

    sys.stdout.write('***** Run = %d/%d ***** \n'  % (run+1, nb_run))
    sys.stdout.write('Minimizing ...\n')
    sys.stdout.write('Fit {0:03d}/{4:03d}-log10(D)={1:+09.4f}-Valid={2:b}-LCC={5:.6f},{6:.6f},{7:.6f},{8:.6f}\n'.format(fit+1, np.log10(distance), valid, run, nb_fit,LCC_results[0], LCC_results[1],LCC_results[2],LCC_results[3]))
    prm = _update_prm(prm_array, prm_user)
    sys.stdout.write(str(prm)+'\n')
    for i in additional_messages:
        sys.stdout.write(i+'\n')
    sys.stdout.flush()

def _callback_random_scan(run, nb_run, loop, loops, distance, valid, prm_array, prm_user):

    os.system('cls' if os.name == 'nt' else 'clear')

    sys.stdout.write('***** Run = %d/%d ***** \n'  % (run+1, nb_run))
    sys.stdout.write('Random Scan...\n')
    sys.stdout.write('Fit {0:06d}/{1:06d}-log10(D)={2:+09.4f}-Valid={3:b}\n'.format(loop, loops, np.log10(distance), valid))
    prm = _update_prm(prm_array, prm_user)
    sys.stdout.write(str(prm))
    sys.stdout.flush()
    

def run_fit(circuit, nb_run, nb_fit, init_types, f_limits, datafilepath, prmfilepath,\
            immittance_type = 'Z',\
            root = './', alloy='Unspecified_Alloy', alloy_id='Unspecified_ID',\
            random_loops=DEFAULT_RANDOM_LOOPS, process_id=1, simplified=False,\
            maxiter=DEFAULT_MAXITER_PER_PARAMETER, maxfun=DEFAULT_MAXFUN_PER_PARAMETER, xtol=DEFAULT_XTOL, ftol=DEFAULT_FTOL,\
            full_output=True, retall=False, disp=False, fmin_callback=None, callback=None):

    #Symbolic Immittance
    I = cdp.get_symbolic_immittance(circuit, immittance_type = immittance_type, simplified = simplified)
    I_num = cdp.get_numeric_immittance(I)
    
    #import parameters
    prmfilepath = os.path.abspath(prmfilepath)
    prm_user = _import_prm_file(prmfilepath)
    _check_parameter_names(prm_user, I)
    _check_parameter_values(prm_user)
    
    prm_init = _initialize_prm_from_immittance(I)
    prm_array = _initialize_prm_from_immittance(I)
    prm_min_run = _initialize_prm_from_immittance(I)
    prm_end_run = _initialize_prm_from_immittance(I)
    
    prm_array = _update_prm(prm_user, prm_array)
    prm_init = _update_prm(prm_user, prm_init)
    prm_min_run = _update_prm(prm_user, prm_min_run)
    prm_end_min = _update_prm(prm_user, prm_end_run)

    mask_to_fit = _get_mask_to_fit(prm_array)
    N = mask_to_fit.size
    
    #import data
    datafilepath = os.path.abspath(datafilepath)
    datafilename, ext = os.path.basename(datafilepath).split('.')
    f, ReZ, ImZ = _get_exp_data(datafilepath)
    w = 2*np.pi*f
    I_exp_complex = ReZ+1j*ImZ
    f_start, f_end = f_limits
    mask, = np.where( (f>=f_start) & (f<=f_end) )

    init_type_0, init_type_validation, init_type_N = init_types

    alloy = alloy.replace(' ','_')
    alloy_id = alloy_id.replace(' ','_')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')

    if len(alloy) == 0:
        alloy = 'Unspecified_Alloy'
    if len(alloy_id) == 0:
        alloy_id = 'Unspecified_ID'

    
    alloy_folder = root + '/' + alloy + '-' + alloy_id
    circuit_str = circuit.replace('+','_s_').replace('/','_p_')
    fit_folder = alloy_folder + '/' + timestamp + '-' + alloy + '-' + alloy_id + '-' + circuit_str + '-' +\
                 init_type_0[0].capitalize() + init_type_validation[0].capitalize() + init_type_N[0].capitalize()+\
                 '-' + '{0:.0e}Hz_{1:.0e}Hz'.format(f_start, f_end)
    alloy_folder = os.path.abspath(alloy_folder)
    fit_folder = os.path.abspath(fit_folder)

    if not os.path.exists(alloy_folder):
        os.mkdir(alloy_folder)       
    os.mkdir(fit_folder)


    filepath = fit_folder + '/' + datafilename + '.' + ext
    filepath = os.path.abspath(filepath)
    shutil.copy(datafilepath, filepath)

    filepath = fit_folder + '/' + datafilename + '-' + circuit_str + '.' + PRM_INIT_EXT
    filepath = os.path.abspath(filepath)
    print filepath
    np.savetxt(filepath, prm_user, fmt=PRM_FORMATS_STR, delimiter='\t', newline='\n', header='\t'.join(PRM_NAMES))
    
    header_minimization_results = '\t'.join(HEADER_MINIMIZATION_ELEMENTS) + '\t'
    header_minimization_results += '\t'.join(prm_user['Names'])
    col = len(HEADER_MINIMIZATION_ELEMENTS) + prm_user['Names'].size
    
    minimization_results = np.zeros(shape=(nb_fit, col), dtype=np.float64)
    
    fit_settings = ['Circuit={0:s}'.format(circuit),\
                    'Immittance Type={0:s}'.format(immittance_type),\
                    'Experimental Data File={0:s}'.format(datafilepath),\
                    'Frequency Range (Hz)={0:.2e},{1:.2e}'.format(f_start, f_end),\
                    'Parameter File={0:s}'.format(prmfilepath),\
                    'No of Processes='.format(str(1)),\
                    'No of Runs per Process='.format(str(nb_run)),\
                    'No of Runs='.format(str(nb_run)),\
                    'No of Fits per Run='.format(str(nb_fit)),\
                    'Initialization Run 1={0:s}'.format(init_type_0),\
                    'Random Loops={0:d}'.format(random_loops),\
                    'Propagation Run > 1={0:s}'.format(init_type_N),\
                    'Initialize after non valid parameters={0:s}'.format(init_type_validation),\
                    'log10 xtol={0:.0f}'.format(np.log10(xtol)),\
                    'log10 ftol={0:.0f}'.format(np.log10(ftol)),\
                    'Iterations/Parameter={0:s}'.format(str(maxiter)),\
                    'fcalls/Parameter={0:s}'.format(str(maxfun)),\
                    'Result Folder={0:s}'.format(os.path.abspath(fit_folder))]
                    
    text = '\n'.join(fit_settings)
    filepath = fit_folder + '/' + datafilename + '-' + circuit_str + '.' + FIT_SETTING_EXT
    filepath = os.path.abspath(filepath)
    fobj = open(filepath,'w')
    fobj.write(text)
    fobj.close()

    
    for run in range(nb_run):
        
        if run == 0:

            if init_type_0 == 'random':
                prm_array = _random_scan(w, prm_array, prm_user, I_exp_complex, I, I_num, run, nb_run, loops=random_loops,\
                                         callback=_callback_random_scan)
            elif init_type_0 == 'user':
                prm_array[:] = prm_init[:]
        else:
            if init_type_N == 'random':
                #prm_array = _get_random_prm_values(prm_array, all_parameters=True)
                prm_array = _random_scan(w, prm_array, prm_user, I_exp_complex, I, I_num, run, nb_run, loops=random_loops,\
                                         callback=_callback_random_scan)
            elif init_type_N == 'user':
                prm_array[:] = prm_init[:]
            elif init_type_N == 'min':
                prm_array[:] = prm_min_run[:]
            elif init_type_N == 'end':
                prm_array[:] = prm_end_run[:]

        
        I_calc_complex = I_num(prm_array['Values'], w)
        distance = _get_distance(I_exp_complex, I_calc_complex)
        distance_min_run = distance
        distance_end_run = distance

        for fit in range(nb_fit):
            prm_array, distance = _minimize(w=w[mask], I_exp_complex=I_exp_complex[mask], I_numeric=I_num, prm_array=prm_array,\
                                            maxiter=maxiter*N, maxfun=maxfun*N, xtol=xtol, ftol=ftol,\
                                            full_output=full_output, retall=retall, disp=disp,callback=fmin_callback)

            prm_output = _update_prm(prm_array, prm_user)
            I_calc_complex = I_num(prm_array['Values'], w)
            LCC_results = _get_LCC(I_exp_complex[mask], I_calc_complex[mask])
            valid = _check_validity_prm(prm_array)
            minimization_results[fit] = np.hstack( (fit+1,\
                                                    int(valid),\
                                                    np.log10(distance),\
                                                    LCC_results,\
                                                    prm_output['Values']) )

            prm_end_run[:] = prm_array[:]
            distance_end_run = distance


            if callback is not None:
                callback(run, nb_run, fit, nb_fit,\
                          distance, valid,\
                          LCC_results, prm_array, prm_user)

            if valid == False:
                if init_type_validation == 'random':
                    prm_array = _get_random_prm_values(prm_array, all_parameters=False)
                elif init_type_validation == 'user':
                    prm_array[:] = prm_init[:]

            elif valid == True:
                if distance_min_run > distance:
                    distance_min_run = distance
                    prm_min_run[:] = prm_array[:]

        if callback is not None:
            callback(run, nb_run, fit, nb_fit,\
                     distance, valid,\
                     LCC_results, prm_end_run, prm_user, additional_messages=['Saving Results ...'])
        _save_results(run, process_id, fit_folder, datafilepath, circuit_str, f, mask, I_exp_complex, I_num,\
                  prm_user, prm_min_run, prm_end_run, distance_min_run, distance_end_run,\
                  minimization_results, header_minimization_results)

    if callback is not None:
        callback(run, nb_run, fit, nb_fit,\
                 distance, valid,\
                 LCC_results, prm_end_run, prm_user, additional_messages=['Computing Summary ...'])
    _get_summary(fit_folder, I, I_num)
    _plot_summary(fit_folder)

if __name__ == '__main__':

    circuit = 'Rel+Qdl/Rct+Rox/Qox+Wox'
    #circuit = 'Rel+Rct/Qdl'

    I = cdp.get_symbolic_immittance(circuit, immittance_type = 'Z', simplified = False)
    I_num = cdp.get_numeric_immittance(I)
    w = np.logspace(-3,6,100)*2*np.pi

    prmfilepath = './test2.PrmInit'
    datafilepath = './141052-2014_11_24-0900-S1-Zy2-EIS.z'

    prm_user = _import_prm_file(prmfilepath)
    
    _check_parameter_names(prm_user, I)
    _check_parameter_values(prm_user)
    
    prm_array = _initialize_prm_from_immittance(I)
    prm_array = _update_prm(prm_user, prm_array)

    Z= I_num(prm_array['Values'], w)
    
    run_fit(circuit, 3, 50, ('random','random','random'), (5e-3, 10e6), datafilepath,\
            prmfilepath,\
            immittance_type = 'Z',\
            root = './', alloy='test', alloy_id='1',\
            random_loops=1000, process_id=1, simplified=False,\
            maxiter=200, maxfun=200, xtol=1e-30, ftol=1e-30,\
            full_output=True, retall=False, disp=False, fmin_callback=None, callback=_callback_fit)

##    filepath = './test-1/2015_02_22-193052-test-1-Rel_s_Qdl_p_Rct_s_Rox_p_Qox_s_Wox-RRR-5e-03Hz_1e+07Hz/'
##    _get_summary(filepath, I, I_num)
##    _plot_summary(filepath)
