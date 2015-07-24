# -*- coding: utf-8 -*-

"""
Documentation is on progress
"""

import os
import datetime
import shutil
import sys
import platform

import numpy as np
from scipy.optimize import fmin
from scipy.stats import linregress
from scipy.stats import t
from scipy.optimize.slsqp import approx_jacobian
# TODO: Write custom approx_jacobian function for specific needs
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rcParams

from . import circuit_decomposition as cdp
from .errors import ParameterNameError, ParameterNumberError, ParameterValueError, FileTypeError

import prettytable as ptb

rcParams['figure.figsize'] = (8, 6)
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.monospace'] = 'Courier New'
rcParams['mathtext.default'] = 'rm'
rcParams['mathtext.fontset'] = 'stix'
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['figure.subplot.hspace'] = 0.5
rcParams['figure.subplot.wspace'] = 0.5
rcParams['legend.numpoints'] = 1  # the number of points in the legend line
rcParams['legend.fontsize'] = 16
rcParams['legend.markerscale'] = 1  # the relative size of legend markers vs. original
rcParams['lines.linewidth'] = 1
rcParams['lines.markeredgewidth'] = 1
rcParams['lines.markersize'] = 4
rcParams['axes.unicode_minus'] = True
from matplotlib.ticker import AutoMinorLocator

# CONSTANTS
# complex number are stored in two sets of the architecture bits
# on 32 bits real and imaginary parts are in 32 bit float i.e.
# the complex number is considered as 2*32bits=64bits floats
ARCHITECTURE, OS = platform.architecture()
RESULT_FORMATTING = '%+.8e'
FLOAT = np.float32
FLOAT_COMPLEX = np.complex64
if ARCHITECTURE == '64bit':
    RESULT_FORMATTING = '%+.16e'
    FLOAT = np.float64
    FLOAT_COMPLEX = np.complex128

_EPSILON = np.sqrt(np.finfo(FLOAT).eps)

ERROR_FORMATTING = '%+.2e'
SUMMARY_RESULT_FORMATTING = '%+.4e'

PRM_NAMES = ['Names',
             'Values',
             'Errors',
             'Min',
             'Max',
             'Fixed',
             'LogRandomize',
             'LogScan',
             'Sign']
PRM_NAME_ALIAS = ['Names',
                  'Values',
                  'Errors',
                  'Min',
                  'Max',
                  'Fixed',
                  'Log. Rand.',
                  'Log. Scan',
                  'Sign']

# noinspection PyTypeChecker
PRM_FORMATS = [(np.str_, 32)] + 4*[FLOAT] + [np.int8] * 4
PRM_FORMATS_STR = ['%s'] + [RESULT_FORMATTING] + [ERROR_FORMATTING] + ['%+.2e'] * 2 + 4 * ['%d']
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

HEADER_RESULT_ARRAY = [r'f /Hz',
                       r'|I_exp| /A', r'Phase_exp /deg', r'Re I_exp /A', r'Im I_exp /A',
                       r'|I_calc| /A', r'Phase_calc /deg', r'Re I_calc /A', r'Im I_calc /A',
                       r'Res |I| /A', r'Res Phase /deg', r'Res Re I /A', r'Res Im I /A']

HEADER_MINIMIZATION_ELEMENTS = ['Nb of Run',
                                'Valid',
                                'log10(D)',
                                'LCC Module',
                                'LCC Phase',
                                'LCC Re',
                                'LCC Im',
                                'slope Module',
                                'slope Phase',
                                'slope Re',
                                'slope Im',
                                'intercept Module',
                                'intercept Phase',
                                'intercept Re',
                                'intercept Im']


def _tanh(z):
    # TODO Look up for Numpy 1.9.3, tanh issue for large complex number should be fixed
    """
    Fix Numpy issue for large complex number.
    Should be fixed in Numpy 1.9.3
    This implementation should overflow for large -negative complex number
    As EIS always uses positive angular frequencies, this implementation is a fairly good work around.
    """
    return (1-np.exp(-2*z))/(1+np.exp(-2*z))

# Shadowing the built-in function
# Avoid overflowing for large complex numbers
np.tanh = _tanh


def _get_header_footer_par_file(filepath):
    """
    
    Parse .par file and get the number of lines for the header and the footer.
    
    Parameters
    ----------
    filepath: str
        Absolute filepath of the .par file.
        
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
    header_flag = True
    footer_flag = False

    with open(filepath, 'r') as fobj:
        for line in fobj:
            nblines += 1
            if line.startswith('<Segment1>'):
                header_flag = False
                skip_header_counter += 1
            elif line.startswith('</Segment1>'):
                footer_flag = True
                skip_footer_counter += 1
            elif header_flag:
                skip_header_counter += 1
            elif footer_flag:
                skip_footer_counter += 1

    skip_header_counter += 3
    nbpoints = nblines - skip_footer_counter - skip_header_counter

    return skip_header_counter, skip_footer_counter, nbpoints


# noinspection PyTypeChecker
def _get_exp_data(filepath):
    r"""

    Get the data array of data files according to their extension.

    Supported files are .z files recorded by ZView software, .par files recorded by VersaStudio and .data files
    were the first three columns represent :math:`f`, :math:`ReZ`, :math:`ImZ`.

    Frequencies, real and imaginary parts of the immittance are extracted from the files.
    
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
    extensions = ['z', 'par', 'data', 'partxt']

    skip_header, skip_footer = 0, 0
    usecols = (0, 1, 2)
    delimiter = '\t'
    converters = None

    if ext.lower() in extensions:

        if ext.lower() == 'par':
            skip_header, skip_footer, nbpoints = _get_header_footer_par_file(filepath)
            usecols = (9, 14, 15)
            delimiter = ','
            converters = None

        elif ext.lower() == 'data':
            skip_header, skip_footer = 0, 0
            usecols = (0, 1, 2)
            delimiter = '\t'
            converters = None

        elif ext.lower() == 'z':
            skip_header, skip_footer, nbpoints = 11, 0, 0
            usecols = (0, 4, 5)
            delimiter = ','
            converters = None

        if ext.lower() == 'partxt':
            skip_header, skip_footer, nbpoints = 0, 0, 0
            usecols = (4, 6, 7)
            delimiter = '\t'
            converters = None

        f, rez, imz = np.genfromtxt(fname=filepath, dtype=FLOAT, comments='#', delimiter=delimiter,
                                    skiprows=0, skip_header=skip_header, skip_footer=skip_footer,
                                    converters=converters, missing='', missing_values=None, filling_values=None,
                                    usecols=usecols, names=None, excludelist=None,
                                    deletechars=None, replace_space='_', autostrip=False, case_sensitive=True,
                                    defaultfmt='f%i', unpack=True, usemask=False, loose=True, invalid_raise=True)

        return f, rez, imz
    else:
        message = 'Data file type was not recognized.'
        raise FileTypeError(message)


def _get_frequency_mask(f, f_limits):
    r"""

    Get the index mask of the frequencies that will be used for computing or fitting EIS data

    Parameters
    -----------
    f: 1d numpy array of floats
        Frequency vector.

    f_limits: tuple of floats
        Start and end values for frequency vector.

    Returns
    --------
    mask: 1d numpy array
        Contains indexes of the frequency vector to be used.
    

    """

    f_start, f_end = f_limits
    mask, = np.where((f >= f_start) & (f <= f_end))

    return mask


def _get_circuit_from_prm(filepath):
    r"""

    Get the electrical circuit from the parameter file.

    Parameters
    -----------
    filepath: string
        Path to the parameter file.

    Returns
    --------
    circuit: string
        String representation of the electrical circuit.

    """
    with open(filepath, 'r') as fobj:
        circuit = fobj.readline().replace(' ', '').replace('\n', '').replace('#', '')

    return circuit


def _import_prm_file(filepath):
    r"""

    Import parameter file which can have three type of extensions: .PrmInit, .PrmEnd, .PrmMin.

    Parameters
    -----------
    filepath: string
        Path to the parameter file.

    Returns
    --------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    """

    dtypes = np.dtype({'names': PRM_NAMES, 'formats': PRM_FORMATS})

    prmfilepath = os.path.abspath(filepath)
    prm_array = np.loadtxt(prmfilepath,
                           comments='#',
                           delimiter='\t',
                           dtype=dtypes,
                           unpack=False)

    # In Python 3, byte and str are two distinct types
    # np.loadtxt imports byte representation of the name column which looks like "b'Name'"
    # --> should be fixed in future numpy release 1.10 (numpy website)
    # TODO Look up the release note of the next numpy release
    # work around consists of replacing b and single quotes
    if sys.version_info[0] == 3:
        for i, in np.ndindex(prm_array.shape):
            prm_array['Names'][i] = prm_array['Names'][i].replace('b', '').replace('\'', '')

    return prm_array


# noinspection PyProtectedMember
def _check_parameter_names(prm_array, symbolic_immittance):
    r"""

    Check if the number and the names of the parameter given in the parameter array correspond
    to the parameters returned by the symbolic expression of the immittance.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    symbolic_immittance: sympy expression
        Symbolic expression of the immittance computed with sympy in the `circuit_decomposition`.

    """

    parameters, parameters_names = cdp._get_parameters(symbolic_immittance)

    nb_parameters = len(parameters)
    nb_parameters_input = len(prm_array['Names'])

    if nb_parameters != nb_parameters_input:
        message = 'The number of given parameters is not equal to the number of parameters ' \
                  'in the expression of the immittance.'
        raise ParameterNumberError(message)

    for name in parameters_names:
        if name not in prm_array['Names']:
            message = 'Imported parameter names do not correspond to the parameter in the expression of the immittance.'
            raise ParameterNameError(message)


def _check_parameter_values(prm_array):
    r"""

    Check if the parameter values are different from zero.

    Check if the parameter lower and upper bounds are different from zero.

    Check if the parameter lower bounds are lower than the upper bounds.

    Check if the options Fixed, LogRandomize, LogScan and Sign are set to -1, 0, or 1.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    Returns
    --------
    None


    """

    mask, = np.where(prm_array['Values'] == 0)
    if mask.size > 0:
        message = 'Parameter values must be different from 0.'
        raise ParameterValueError(message)

    mask, = np.where(prm_array['Min'] == 0.0)
    if mask.size > 0:
        message = 'Lower bounds must be different from 0.'
        raise ParameterValueError(message)

    mask, = np.where(prm_array['Max'] == 0.0)
    if mask.size > 0:
        message = 'Upper bounds must be different from 0.'
        raise ParameterValueError(message)

    mask, = np.where(prm_array['Min'] >= prm_array['Max'])
    if mask.size > 0:
        message = 'Lower bounds cannot be higher than upper bounds.'
        raise ParameterValueError(message)

    mask, = np.where((prm_array['Fixed'] != 1) & (prm_array['Fixed'] != 0))
    if mask.size > 0:
        message = 'Fixed option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)

    mask, = np.where((prm_array['LogRandomize'] != 1) & (prm_array['LogRandomize'] != 0))
    if mask.size > 0:
        message = 'LogRandomize option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)

    mask, = np.where((prm_array['LogScan'] != 1) & (prm_array['LogScan'] != 0))
    if mask.size > 0:
        message = 'LogScan option can only be True or False i.e. 1 or 0.'
        raise ParameterValueError(message)

    mask, = np.where((prm_array['Sign'] != 1) & (prm_array['Sign'] != 0) & (prm_array['Sign'] != -1))
    if mask.size > 0:
        message = 'Sign option can only be positive, negative or both i.e. +1, 0 or -1.'
        raise ParameterValueError(message)


def _get_mask_to_fit(prm_array):
    r"""

    Get the indexes of the parameters to be fitted.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    Returns
    --------
    mask: 1d numpy array
        Contains indexes of the parameters to be fitted.

    """

    mask, = np.where(prm_array['Fixed'] == 0)
    return mask


def _get_mask_not_valid(prm_array):
    r"""

    Get the indexes of the parameters that are not valid i.e. the values are out of the bounds.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    Returns
    --------
    mask: 1d numpy array
        Contains indexes of the parameters that are not valid.

    """

    mask_to_fit = _get_mask_to_fit(prm_array)

    values = prm_array['Values'][mask_to_fit]
    min_value = prm_array['Min'][mask_to_fit]
    max_value = prm_array['Max'][mask_to_fit]

    mask, = np.where((values > max_value) | (values < min_value))

    return mask


def _get_mask_logscan(prm_array):
    r"""

    Get the indexes of the parameters to be scanned in the logarithmic scale.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    Returns
    --------
    mask: 1d numpy array
        Contains indexes of the parameters to be scanned in the logarithmic scale.

    """

    mask_to_fit = _get_mask_to_fit(prm_array)
    mask, = np.where(prm_array['LogScan'][mask_to_fit] == 1)

    return mask


def _check_validity_prm(prm_array):
    r"""

    Check if there are parameters that are not valid.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    Returns
    --------
    valid: bool
        Flag indicating if there are parameters that are not valid.
        

    """

    valid = False

    mask_not_valid = _get_mask_not_valid(prm_array)

    if mask_not_valid.size == 0:
        valid = True

    return valid


# noinspection PyProtectedMember
def _initialize_prm_from_immittance(symbolic_immittance):
    r"""

    Initialize a new parameter array from the symbolic expression of the immittance.

    Parameters
    -----------
    symbolic_immittance: sympy expression
        Symbolic expression of the immittance computed with sympy in the `circuit_decomposition`.

    Returns
    --------
    prm_array: 2d numpy array
        Parameter array initialized with zeros.

    """

    parameters, parameter_names = cdp._get_parameters(symbolic_immittance)
    dtypes = np.dtype({'names': PRM_NAMES, 'formats': PRM_FORMATS})
    prm_array = np.zeros(shape=(len(parameters),), dtype=dtypes)
    prm_array['Names'] = parameter_names

    return prm_array


def _update_prm(from_prm, to_prm):
    r"""

    Parameter order defined in the parameter files can be different from the order of the parameters
    given by the symbolic expression of the immittance.

    Update the `to_prm` array values from the `from_prm array`. 


    Parameters
    -----------
    from_prm: 2d numpy array
        Parameter array.

    to_prm: 2d numpy array
        Parameter array.

    Returns
    --------
    to_prm: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    """

    for ind, name in enumerate(to_prm['Names']):
        mask, = np.where(from_prm['Names'] == name)
        to_prm[ind] = from_prm[mask]

    return to_prm


def _get_random_prm_values(prm_array, all_parameters=False):
    r"""

    Generate random values for the parameters that are not fixed. The randomization can be applied to all
    or only to the invalid parameters.

    Parameters
    -----------
    prm_array: 2d numpy array
        Parameter array containing all necessary information of the parameters.

    all_parameters: bool
        Flag indicating if all not fixed parameters must be randomized no matter if there are valid or not.

    Returns
    --------
    prm_array: 2d numpy array
        Parameter array with randomized values.

    """

    mask_to_fit = _get_mask_to_fit(prm_array)

    # if mask_not_valid is an empty array
    # no change to prm_array
    # initial randomizing
    if not all_parameters:
        mask_not_valid = _get_mask_not_valid(prm_array)
    else:
        mask_not_valid = np.arange(mask_to_fit.size)

    # two sub 1d arrays 'values_to_fit' and 'Values' from prm_array field 'Values' are
    # necessary for updating values after calculations
    # Min, Max and LogScan do not need subarrays -> only accessing values
    values_to_fit = prm_array['Values'][mask_to_fit]
    value = values_to_fit[mask_not_valid]
    lbounds = prm_array['Min'][mask_to_fit][mask_not_valid]
    ubounds = prm_array['Max'][mask_to_fit][mask_not_valid]
    lograndomize = prm_array['LogRandomize'][mask_to_fit][mask_not_valid]
    sign = prm_array['Sign'][mask_to_fit][mask_not_valid]

    mask_linear, = np.where(lograndomize == 0)
    mask_log, = np.where(lograndomize == 1)
    mask_positive = np.where(sign == 1)
    mask_negative = np.where(sign == -1)
    mask_both = np.where(sign == 0)
    random_sign = np.random.randint(-1, 1, len(mask_both))
    random_sign[random_sign == 0] = 1.0

    # in linear scale the random values are classically calculated by: value = low + random(0,1)*(up-low)
    # in log scale the random values are calculated using the logarithmic values of the limits:
    # value = 10**( log10(low) + random(0,1)*(log10(up)-log(low)) )
    value[mask_linear] = lbounds[mask_linear] + np.random.random((mask_linear.size,)) * (
        ubounds[mask_linear] - lbounds[mask_linear])
    value[mask_log] = 10 ** (np.log10(lbounds[mask_log]) + np.random.random((mask_log.size,)) * (
        np.log10(ubounds[mask_log]) - np.log10(lbounds[mask_log])))

    value[mask_positive] *= 1.0
    value[mask_negative] *= -1.0
    value[mask_both] *= random_sign

    values_to_fit[mask_not_valid] = value
    prm_array['Values'][mask_to_fit] = values_to_fit

    return prm_array


def _get_distance(immittance_exp, immittance_calc, weights=None):
    r"""

    Compute the distance :math:`D` between the experimental immittance :math:`I_{exp}` and
    the calculated immittance :math:`I_{calc}`.

    The weights are set to the inverse of the modulus of the experimental immittance
    :math:`1/\left | I_{exp} \right |^{2}`.

    .. math::
        
        \Delta Re & = Re \, I_{exp} - Re \, I_{calc} \\
        \Delta Im & = Im \, I_{exp} - Im \, I_{calc} \\
        D & = \sum weights \cdot (\Delta Re^2 + \Delta Im^2)

    Parameters
    ----------
    I_exp: 1d numpy array
        Contains the complex values of the :math:`I_{exp}`.

    I_calc: 1d numpy array
        Contains the complex values of the :math:`I_{calc}`.

    weights: 1d numpy array, optional
        Weights to be used in the computation of the distance. By default, the experimental module is used.

    Returns
    -------
    D: float
        The computed distance :math:`D` on real and imaginary parts of :math:`I`:.

    """
    mod_immittance_exp = np.absolute(immittance_exp)
    re_immittance_exp = np.real(immittance_exp)
    im_immittance_exp = np.imag(immittance_exp)

    re_immittance_calc = np.real(immittance_calc)
    im_immittance_calc = np.imag(immittance_calc)

    delta_re = (re_immittance_exp - re_immittance_calc)
    delta_im = (im_immittance_exp - im_immittance_calc)

    if weights is None:
        weights = 1.0 / mod_immittance_exp ** 2

    distance = np.sum(weights * (delta_re ** 2 + delta_im ** 2))

    return distance


def _get_residuals(p, w, immittance_exp, immittance_num, weights=None):
    r"""
    Compute the weighted module of the residuals between calculated and experimental values.

    Parameters
    ----------
    p: 1d numpy array
        Contains the parameter values.

    w: 1d numpy array
        Angular frequencies for which the complex values of :math:`I` have to be calculated.

    immittance_exp: 1d numpy array
        Contains the complex values of the experimental immittance :math:`I`.

    immittance_num: numpy ufunc
        Lambdified immittance from the symbolic expression.

    weights: 1d numpy array, optional
        Weights to be used in the computation of the distance. By default, the experimental immittance is used.

    Returns
    -------
    residuals: 1d numpy array
        Contains the residuals for each angular frequency.
    """
    if weights is None:
        weights = 1.0/immittance_exp
    return np.absolute((immittance_num(p, w) - immittance_exp)*weights)


def _get_chi2(p, w, immittance_exp, immittance_num, weights=None):
    r"""
    Compute the scalar :math:`\chi ^{2}` by computing the weighted module of the residuals between
    calculated and experimental values.

    .. math::

        \chi ^{2} & = \sum \epsilon_{i}^{2} \\
        \chi ^{2} & = \sum w_i^2 \vert I(p_k , \omega _{i}) - Iexp_i \vert ^2

    Parameters
    ----------
    p: 1d numpy array
        Contains the parameter values.

    w: 1d numpy array
        Angular frequencies for which the complex values of :math:`I` have to be calculated.

    immittance_exp: 1d numpy array
        Contains the complex values of the experimental immittance :math:`I`.

    immittance_num: numpy ufunc
        Lambdified immittance from the symbolic expression.

    weights: 1d numpy array, optional
        Weights to be used in the computation of the distance. By default, the experimental module is used.

    Returns
    -------
   :math:`\chi ^{2}`: 1d numpy array
        Scalar :math:`\chi ^{2}`.
    """
    return np.sum(_get_residuals(p, w, immittance_exp, immittance_num, weights=weights)**2)


def _target_function(p, w, prm_array, immittance_exp, immittance_num):
    r"""

    Update the prm_array with the vector `p` sent by the fmin procedure (see Scipy for more details).

    The calculated complex values of :math:`I` will be sent along the experimental values
    to the :func:`_get_distance` function.

    The value of the distance between the experimental and calculated data will sent back to the optimization algorithm.
    

    Parameters
    ----------
    p: 1d array
        Parameter vector sent by the optimization algorithm.

    w: 1d array
        Angular frequencies for which the complex values of :math:`I` have to be calculated.

    prm_array: 2d array
        Parameter array containing all necessary information of the parameters.

    immittance_exp: 1d numpy array
        Contains the complex values of the experimental immittance :math:`I`.

    immittance_num: numpy ufunc
        Lambdified immittance from the symbolic expression.
    
    Returns
    -------
    distance: float
        Calculated distance between experimental and calculated data values.
        See the :func:`_get_distance` function.

    """

    mask_to_fit = _get_mask_to_fit(prm_array)
    mask_logscan = _get_mask_logscan(prm_array)

    p0 = prm_array['Values'][mask_to_fit]
    p0[:] = p[:]
    p0[mask_logscan] = 10 ** p0[mask_logscan]
    prm_array['Values'][mask_to_fit] = p0

    # immittance_calc = immittance_num(prm_array['Values'], w)
    # distance = _get_distance(immittance_exp, immittance_calc)

    return _get_chi2(prm_array['Values'], w, immittance_exp, immittance_num, weights=None)


def _minimize(w, immittance_exp_complex, immittance_num, prm_array,
              maxiter=None, maxfun=None, xtol=DEFAULT_XTOL, ftol=DEFAULT_FTOL,
              full_output=True, retall=False, disp=False, callback=None):
    r"""

    Execute the Nelder-Mead algorithm through the `fmin` procedure (see Scipy documentation) based
    on parameter values given by ``prm_array`` and the angular frequency vector :math:`\omega`.

    Parameters
    ----------
    w: 1d array
        Angular frequencies.

    I_exp_complex: 1d numpy array
        Contains the complex values of the experimental immittance :math:`I`.

    I_numeric: numpy ufunc
        Lambdified immittance from the symbolic expression.

    prm_array: 2d array
        Parameter array containing all necessary information of the parameters.

    maxiter : int, optional
        Maximum number of iterations to perform.

    maxfun : number, optional
        Maximum number of function evaluations to make.

    xtol : float, optional
        Relative error in xopt acceptable for convergence.
        
    ftol : number, optional
        Relative error in func(xopt) acceptable for convergence.

    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.

    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        
    disp : bool, optional
        Set to True to print convergence messages.

    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    prm_array: 2d array
        Parameter array containing the updated values of the parameters.
        
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
        
    """

    mask_to_fit = _get_mask_to_fit(prm_array)
    mask_logscan = _get_mask_logscan(prm_array)
    p0 = prm_array['Values'][mask_to_fit]
    p0[mask_logscan] = np.log10(p0[mask_logscan])

    popt, fopt, iteration, funcalls, warnflag = fmin(_target_function,
                                                     p0,
                                                     args=(w,
                                                           prm_array,
                                                           immittance_exp_complex,
                                                           immittance_num),
                                                     maxiter=maxiter,
                                                     maxfun=maxfun,
                                                     xtol=xtol,
                                                     ftol=ftol,
                                                     full_output=full_output,
                                                     retall=retall,
                                                     disp=disp,
                                                     callback=callback)

    p0 = prm_array['Values'][mask_to_fit]
    p0[:] = popt[:]
    p0[mask_logscan] = 10 ** p0[mask_logscan]
    prm_array['Values'][mask_to_fit] = p0

    return prm_array, fopt


def _get_complex_parameters(z, deg=True):
    r"""

    Get the modulus, the phase, the real and imaginary parts from a complex array.

    Parameters
    -----------
    z: 1d numpy array
        Vector of complex number to be processed.

    Returns
    --------
    mod: 1d numpy array
        Modulus of the z vector

    phase: 1d numpy array
        Phase of the z vector

    Re: 1d numpy array
        Real part of the z vector.

    Im: 1d numpy array
        Imaginary part of the z vector.

    """

    mod = np.absolute(z)
    phase = np.angle(z, deg=deg)
    re = np.real(z)
    im = np.imag(z)

    return mod, phase, re, im


# noinspection PyPep8
def _get_lcc(immittance_exp_complex, immittance_calc_complex):
    r"""

    Compute the correlation coefficients, the slope and the intercept between experimental and calculated
    values for the modulus, phase, real and imaginary parts.

    The computation is performed through the `linregress` procedure (see Scipy for more details).

    Parameters
    -----------
    I_exp_complex: 1d numpy array
        Contains the complex values of the experimental immittance :math:`I`.

    I_calc_complex: 1d numpy array
        Contains the complex values of the calculated immittance :math:`I`.

    Returns
    --------
    r_mod: float
        Correlation coefficient for the modulus.

    r_phase: float
        Correlation coefficient for the phase.

    r_Re: float
        Correlation coefficient for the real part.

    r_Im: float
        Correlation coefficient for the imagianry part.

    slope_mod: float
        Slope for the modulus.

    slope_phase: float
        Slope for the phase.

    slope_Re: float
        Slope for the real part.

    slope_Im: float
        Slope for the imagianry part.

    intercept_mod: float
        Intercept for the modulus.

    intercept_phase: float
        Intercept for the phase.

    intercept_Re: float
        Intercept for the real part.

    intercept_Im: float
        Intercept for the imaginary part.

    """

    mod_exp, phase_exp, re_exp, im_exp = _get_complex_parameters(immittance_exp_complex, deg=True)
    mod_calc, phase_calc, re_calc, im_calc = _get_complex_parameters(immittance_calc_complex, deg=True)

    slope_mod, intercept_mod, r_mod, p_mod, std_mod = linregress(mod_exp, mod_calc)
    slope_phase, intercept_phase, r_phase, p_phase, std_phase = linregress(phase_exp, phase_calc)
    slope_re, intercept_re, r_re, p_re, std_re = linregress(re_exp, re_calc)
    slope_im, intercept_im, r_im, p_im, std_im = linregress(im_exp, im_calc)

    return r_mod, r_phase, r_re, r_im, \
           slope_mod, slope_phase, slope_re, slope_im, \
           intercept_mod, intercept_phase, intercept_re, intercept_im


def _get_results_array(f, immittance_exp_complex, immittance_calc_complex):
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
    data_array: 2d numpy array
        Array containing the experimental and calculated values.

    """
    header = '\t'.join(HEADER_RESULT_ARRAY)

    mod_exp, phase_exp, re_exp, im_exp = _get_complex_parameters(immittance_exp_complex, deg=True)
    mod_calc, phase_calc, re_calc, im_calc = _get_complex_parameters(immittance_calc_complex, deg=True)

    res_mod = mod_exp - mod_calc
    res_phase = phase_exp - phase_calc
    res_re = re_exp - re_calc
    res_im = im_exp - im_calc

    data_array = np.transpose(np.vstack((f,
                                         mod_exp, phase_exp, re_exp, im_exp,
                                         mod_calc, phase_calc, re_calc, im_calc,
                                         res_mod, res_phase, res_re, res_im)))

    return header, data_array


# noinspection PyUnresolvedReferences
def _save_results(circuit, run, process_id, fit_folder, datafilepath, circuit_str, f, mask,
                  immittance_exp_complex, immittance_num,
                  prm_user, prm_min_run, prm_end_run, distance_min_run, distance_end_run,
                  minimization_results, header_minimization_results):
    r"""

    Save the minimum and the end parameter array as well as the result arrays computed from them.

    Parameters
    -----------
    run: int
        Indicates the current run.

    process_id: int
        Indicates the current process if the parallel are processes are used.

    fit_folder: string
        Path the fit folder where the files will be saved.

    datafilepath: string
        Filepath to the experimental data file.

    circuit_str: string
        Circuit representation to be added in the naming of the result files.

    f: 1d numpy array
        Frequency vector.

    mask: 1d numpy array
        Contains indexes of the frequency vector to be used.

    I_exp_complex:
        Contains the complex values of :math:`I_{exp}`.
        
    I_numeric: numpy ufunc
        Lambdified immittance from the symbolic expression.

    prm_user: 1d numpy array
        Parameter array containing all necessary information of the parameters.

    prm_min_run:
        Parameter array for the minimal distance of the minimizations. 

    prm_end_run:
        Parameter array for the end distance of the minimizations. 
    
    distance_min_run:
        Minimal distance achieved during the minimization. 

    distance_end_run:
        End distance achieved during the minimization. 

    minimization_results:
        All values for each minimization.

    header_minimization_results:
        Header for the `minimization_results` array.

    Returns
    --------
    None


    """

    name, ext = os.path.basename(datafilepath).split('.')
    n = prm_end_run['Names'].size
    w = 2 * np.pi * f

    # save minimization results
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}.{5:s}'.format(fit_folder, name, circuit_str, process_id, run + 1,
                                                            PRM_ALL_RUN_EXT)
    np.savetxt(filepath, minimization_results,
               fmt=['%d', '%d', '%+.4f'] + [RESULT_FORMATTING] * (len(HEADER_MINIMIZATION_ELEMENTS) - 3 + n),
               delimiter='\t',
               newline='\n',
               header=header_minimization_results)

    # save minimum
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run + 1,
                                                                     np.log10(distance_min_run), PRM_MIN_EXT)
    np.savetxt(filepath, _update_prm(prm_min_run, prm_user), fmt=PRM_FORMATS_STR, delimiter='\t', newline='\n',
               header=circuit + '\n' + '\t'.join(PRM_NAMES))

    immittance_calc_complex = immittance_num(prm_min_run['Values'], w)
    header, data = _get_results_array(f[mask], immittance_exp_complex[mask], immittance_calc_complex[mask])
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run + 1,
                                                                     np.log10(distance_min_run), DATA_MIN_EXT)
    np.savetxt(filepath, data, fmt=RESULT_FORMATTING, delimiter='\t', newline='\n', header=header)

    ext = 'pdf'
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-{5:s}-d{6:.4f}.{7:s}'.format(fit_folder, name, circuit_str, process_id,
                                                                           run + 1, 'Min', np.log10(distance_min_run),
                                                                           ext)
    _save_pdf(filepath,
              f, immittance_exp_complex, immittance_calc_complex,
              mask, minimization_results, data)

    # save end
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run + 1,
                                                                     np.log10(distance_end_run), PRM_END_EXT)
    np.savetxt(filepath, _update_prm(prm_end_run, prm_user), fmt=PRM_FORMATS_STR, delimiter='\t', newline='\n',
               header=circuit + '\n' + '\t'.join(PRM_NAMES))

    immittance_calc_complex = immittance_num(prm_end_run['Values'], w)
    header, data = _get_results_array(f[mask], immittance_exp_complex[mask], immittance_calc_complex[mask])
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-d{5:.4f}.{6:s}'.format(fit_folder, name, circuit_str, process_id, run + 1,
                                                                     np.log10(distance_end_run), DATA_END_EXT)
    np.savetxt(filepath, data, fmt=RESULT_FORMATTING, delimiter='\t', newline='\n', header=header)

    ext = 'pdf'
    filepath = '{0:s}/{1:s}-{2:s}-{3:d}-{4:d}-{5:s}-d{6:.4f}.{7:s}'.format(fit_folder, name, circuit_str, process_id,
                                                                           run + 1, 'End', np.log10(distance_end_run),
                                                                           ext)
    _save_pdf(filepath,
              f, immittance_exp_complex, immittance_calc_complex,
              mask, minimization_results, data)


def _save_pdf(filepath,
              f, immittance_exp_complex, immittance_calc_complex,
              mask, minimization_results, data):
    r"""

    Save the plots of the results for the minimum and the final distance after each minimization.

    Nyquist plot, Bode modulus and phase plots as well as the residual plots are saved in one common pdf file.

    Parameters
    -----------
    filepath: string

    f: 1d numpy array

    I_exp_complex:
        Contains the complex values of :math:`I_{exp}`.

    I_calc_complex:
        Contains the complex values of :math:`I_{calc}`.

    mask: 1d numpy array
        Contains indexes of the frequency vector to be used.

    minimization_results:
        All values for each minimization.

    data: 2d numpy array
        Array containing the experimental and calculated values.
        

    Returns
    --------
    None

    """

    pdf = PdfPages(filepath)
    # scilimits = (1e-6,1e6)

    mod_exp, phase_exp, re_exp, im_exp = _get_complex_parameters(immittance_exp_complex, deg=True)
    mod_calc, phase_calc, re_calc, im_calc = _get_complex_parameters(immittance_calc_complex, deg=True)

    # Nyquist Plot - fitting range
    plt.figure(figsize=(8, 6))
    plt.title(r'ReZ vs ImZ (fitting range)')
    plt.grid(which='major', axis='both')
    plt.xlabel(r'ReZ /$\Omega$')
    plt.ylabel(r'ImZ /$\Omega$')
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(re_exp[mask], im_exp[mask], 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(re_calc[mask], im_calc[mask], 'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Nyquist Plot - full range
    plt.figure(figsize=(8, 6))
    plt.title(r'ReZ vs ImZ (full range)')
    plt.grid(which='major', axis='both')
    plt.xlabel(r'ReZ /$\Omega$')
    plt.ylabel(r'ImZ /$\Omega$')
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(re_exp, im_exp, 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(re_calc, im_calc, 'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Phase Plot - fitting range
    plt.figure(figsize=(8, 6))
    plt.title(r'$\theta$ vs f (fitting range)')
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='x')
    plt.xlabel(r'f /Hz')
    plt.ylabel(r'$\theta$ /$^{\circ}$')
    plt.xscale('log')
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(f[mask], phase_exp[mask], 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f[mask], phase_calc[mask], 'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Phase Plot - full range
    plt.figure(figsize=(8, 6))
    plt.title(r'$\theta$ vs f (full range)')
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='x')
    plt.xlabel(r'f /Hz')
    plt.ylabel(r'$\theta$ /$^{\circ}$')
    plt.xscale('log')
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(f, phase_exp, 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f, phase_calc, 'r.-', markersize=4, linewidth=1, label='fit')
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Module Plot - fitting range
    plt.figure(figsize=(8, 6))
    plt.title(r'|Z| vs f (fitting range)')
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='both')
    plt.xlabel(r'f /Hz')
    plt.ylabel('|Z| /$\Omega$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(f[mask], mod_exp[mask], 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f[mask], mod_calc[mask], 'r.-', markersize=4, linewidth=1, label='fit')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Module Plot - full range
    plt.figure(figsize=(8, 6))
    plt.title(r'|Z| vs f (full range)')
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='both')
    plt.xlabel(r'f /Hz')
    plt.ylabel('|Z| /$\Omega$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(f, mod_exp, 'k-o', markersize=4, markeredgewidth=1, mfc='w', mec='k', label='exp')
    plt.plot(f, mod_calc, 'r.-', markersize=4, linewidth=1, label='fit')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # Distance Plot
    plt.figure(figsize=(8, 6))
    mask_, = np.where(minimization_results[:, 0] != 0)
    mask_valid, = np.where(minimization_results[mask_, 1] == 1)
    plt.plot(minimization_results[mask_, 0], minimization_results[mask_, 2], color='k', marker='o', linestyle='-',
             linewidth=1, mfc='w', mec='k', markeredgewidth=1, label='Not Valid')
    plt.plot(minimization_results[mask_valid, 0], minimization_results[mask_valid, 2], color='g', marker='o',
             linestyle='None', linewidth=1, mfc='w', mec='g', markeredgewidth=1, label='Valid')
    plt.title('log10(D) vs no fit')
    plt.grid(which='major', axis='both')
    plt.xlabel('No of minimization')
    plt.ylabel('log(D)')
    plt.legend(loc='best')
    pdf.savefig()
    plt.close()

    # FQ-Plot
    # see B. A. Boukamp, “A package for impedance/admittance data analysis,”
    # Solid State Ionics, vol. 18–19, no. Part 1, pp. 136–140, 1986.
    # dRe/Re_exp, dIm/Im_exp vs f in log
    plt.figure(figsize=(8, 6))
    plt.plot(f[mask], data[:, 11]/mod_exp[mask]*100.0, 'ko',
             markersize=4, markeredgewidth=1, mfc='k', mec='k', label='$\Delta Re$')
    plt.plot(f[mask], data[:, 12]/mod_exp[mask]*100.0, 'ko',
             markersize=4, markeredgewidth=1, mfc='w', mec='k', label='$\Delta Im$')
    plt.title('FQ-Plot')
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='x')
    plt.xscale('log')
    plt.xlabel('f /Hz')
    plt.ylabel(u'Relative Error /%')
    plt.legend(loc='upper center', ncol=2)
    pdf.savefig()
    plt.close()

    # Residuals Re
    plt.figure(figsize=(8, 6))
    plt.title(r'Residuals Re')
    plt.xlabel(r'$ReZ_{calc} - ReZ_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:, 11], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, color='k')
    pdf.savefig()
    plt.close()

    # Residuals Im
    plt.figure(figsize=(8, 6))
    plt.title(r'Residuals Im')
    plt.xlabel(r'$ImZ_{calc} - ImZ_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:, 12], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, color='k')
    pdf.savefig()
    plt.close()

    # Residuals Phase
    plt.figure(figsize=(8, 6))
    plt.title(r'Residuals Phase')
    plt.xlabel(r'$\theta_{calc} - \theta_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:, 10], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, color='k')
    pdf.savefig()
    plt.close()

    # Residuals Module
    plt.figure(figsize=(8, 6))
    plt.title(r'Residuals Module')
    plt.xlabel(r'$|Z|_{calc} - |Z|_{exp}$')
    plt.ylabel('Normalized Frequency')
    plt.hist(data[:, 9], bins=20, normed=True, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False, color='k')
    pdf.savefig()
    plt.close()

    pdf.close()


# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
def _get_summary(fit_folder, symbolic_immittance, numeric_immittance):
    r"""

    List the result files for parameters at the end and the minimum of each run.

    Compute the distance, the LCCs for the frequency range that was used for minimizing the target function.

    The results are saved in 2 files: .SumEnd, .SumMin.


    Parameters
    -----------
    fit_folder: string
        Path of the fit folder.

    symbolic_immittance: sympy expression
        Symbolic expression of the immittance computed with sympy in the `circuit_decomposition`.
    
    numeric_immittance: numpy ufunc
        Lambdified immittance from the symbolic expression.


    Returns
    -------
    None
        
    """

    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)
    prm_end = []
    prm_min = []
    fitsettings_filepath = ''
    prm_array = _initialize_prm_from_immittance(symbolic_immittance)

    run = 0
    for i in listfiles:
        ext = i.split('.')[-1]
        if ext == PRM_END_EXT:
            run += 1
            prm_end.append(os.path.abspath(dirpath + '/' + i))
        elif ext == PRM_MIN_EXT:
            prm_min.append(os.path.abspath(dirpath + '/' + i))
        # elif ext in DATA_FILE_EXTS:
        # datafilepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    fitsettings_fobj = open(fitsettings_filepath, 'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n', '')

    f_start, f_end = fitsettings_dict['Frequency Range (Hz)'].split(',')
    f_start, f_end = float(f_start), float(f_end)

    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    prm_user_filepath = os.path.abspath(fitsettings_dict['Parameter File'])
    circuit = fitsettings_dict['Circuit']
    circuit_str = circuit.replace('+', '_s_').replace('/', '_p_')

    f, rez, imz = _get_exp_data(datafilepath)
    w = 2 * np.pi * f
    immittance_exp_complex = rez + 1j * imz
    mask, = np.where((f >= f_start) & (f <= f_end))

    prm_user = _import_prm_file(prm_user_filepath)
    n = prm_user.size
    header = '\t'.join(HEADER_MINIMIZATION_ELEMENTS) + '\t'
    header += '\t'.join(prm_user['Names'])
    col = len(HEADER_MINIMIZATION_ELEMENTS) + prm_user['Names'].size

    header_dtypes = np.dtype({'names':header.split('\t'),
                                  'formats':[(np.str_, 128)]+[FLOAT]*(len(header.split('\t'))-1)})

    summary_end = np.zeros(shape = (run,), dtype=header_dtypes)
    for ind, i in enumerate(prm_end):
        run_i, minimization_i = os.path.basename(i).split('-d')[0].split('-')[-2:]
        prm = _import_prm_file(os.path.abspath(i))
        prm_array = _update_prm(prm, prm_array)
        valid = _check_validity_prm(prm_array)

        immittance_calc_complex = numeric_immittance(prm_array['Values'], w)
        distance = _get_distance(immittance_exp_complex[mask], immittance_calc_complex[mask])
        lcc_results = _get_lcc(immittance_exp_complex[mask], immittance_calc_complex[mask])

        summary_end[ind] = (run_i + '-' + minimization_i, valid, np.log10(distance)) \
                           + tuple(lcc_results) + tuple(prm['Values'].tolist())
    # sort over the log10(D)
    mask_end = np.argsort(summary_end['log10(D)'])
    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '.' + SUMMARY_END_EXT)
    np.savetxt(filepath, X=summary_end[mask_end],
               fmt=['%s', '%d', '%+.4f'] + [RESULT_FORMATTING] * (len(HEADER_MINIMIZATION_ELEMENTS) - 3 + n),
               delimiter='\t', header=header)

    summary_min = np.zeros(shape = (run,), dtype=header_dtypes)
    for ind, i in enumerate(prm_min):
        run_i, minimization_i = os.path.basename(i).split('-d')[0].split('-')[-2:]
        prm = _import_prm_file(os.path.abspath(i))
        prm_array = _update_prm(prm, prm_array)
        valid = _check_validity_prm(prm_array)

        immittance_calc_complex = numeric_immittance(prm_array['Values'], w)
        distance = _get_distance(immittance_exp_complex[mask], immittance_calc_complex[mask])
        lcc_results = _get_lcc(immittance_exp_complex[mask], immittance_calc_complex[mask])

        summary_min[ind] = (run_i + '-' + minimization_i, valid, np.log10(distance)) \
                           + tuple(lcc_results) + tuple(prm['Values'].tolist())
    # sort over the log10(D)
    mask_min = np.argsort(summary_min['log10(D)'])
    filepath = os.path.abspath(result_folder + '/' + basename + '-' + circuit_str + '.' + SUMMARY_MIN_EXT)
    np.savetxt(filepath, X=summary_min[mask_min],
               fmt=['%s', '%d', '%+.4f'] + [RESULT_FORMATTING] * (len(HEADER_MINIMIZATION_ELEMENTS) - 3 + n),
               delimiter='\t', header=header)


def _plot_summary(fit_folder):
    r"""

    Plot the result files that were created by the :func:`_get_summary` for the parameter values
    at the end and the minimum of each run.

    The results are saved in 2 files: -0-End.pdf, -0-Min.pdf.


    Parameters
    -----------
    fit_folder: string
        Path of the fit folder.


    """

    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)

    summary_end_filepath = ''
    summary_min_filepath = ''
    fitsettings_filepath = ''

    for i in listfiles:
        cuts = i.split('.')
        ext = cuts[-1]
        if ext == SUMMARY_END_EXT:
            summary_end_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == SUMMARY_MIN_EXT:
            summary_min_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    fitsettings_fobj = open(fitsettings_filepath, 'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n', '')

    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    prm_user_filepath = os.path.abspath(fitsettings_dict['Parameter File'])
    circuit = fitsettings_dict['Circuit']
    circuit_str = circuit.replace('+', '_s_').replace('/', '_p_')

    prm_user = _import_prm_file(prm_user_filepath)
    header = '\t'.join(HEADER_MINIMIZATION_ELEMENTS) + '\t'
    header += '\t'.join(prm_user['Names'])
    header_dtypes = np.dtype({'names':header.split('\t'),
                                  'formats':[(np.str_, 128)]+[FLOAT]*(len(header.split('\t'))-1)})
    summary_end = np.loadtxt(summary_end_filepath, comments='#',
                             delimiter='\t',
                             skiprows=0,
                             ndmin=2,
                             unpack=False,
                             dtype=header_dtypes)

    summary_min = np.loadtxt(summary_min_filepath, comments='#',
                             delimiter='\t',
                             skiprows=0,
                             ndmin=2,
                             unpack=False,
                             dtype=header_dtypes)

    filepath = os.path.abspath(
        result_folder + '/' + basename + '-' + circuit_str + '-' + '0' + '-' + 'End' + '.' + 'pdf')
    pdf_end = PdfPages(filepath)
    filepath = os.path.abspath(
        result_folder + '/' + basename + '-' + circuit_str + '-' + '0' + '-' + 'Min' + '.' + 'pdf')
    pdf_min = PdfPages(filepath)

    scilimits = (-4, 4)

    row = summary_end.shape[0]
    no_run = range(1, row+1)
    for ind, name in enumerate(prm_user['Names']):
        if name[0] in ['R', 'D', 'M']:
            unit = '/$\Omega$'
        elif name[0] in ['W']:
            unit = '/$\Omega \cdot s^{-1/2}$'
        elif name[0] in ['C']:
            unit = '/F'
        elif name[0] in ['L']:
            unit = '/H'
        elif name[0] in ['Q']:
            unit = '/$\Omega ^{-1} \cdot s^{n}$'
        elif name[0] in ['n', 'N']:
            unit = ''
        elif name[0] in ['T']:
            unit = 's'
        else:
            unit = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.ticklabel_format(scilimits=scilimits)
        ax.grid()
        ax.set_title('{0:s} vs Run'.format(name))
        ax.set_xlabel('No Run')
        ax.set_ylabel(r'Values {1:s}'.format(name, unit))
        ax.plot(no_run, summary_end[name], color='k', marker='o', mfc='w', mec='k', ls='-', lw=1, ms=4, mew=1)
        ax.set_xticks(no_run)
        ax.set_xticklabels(summary_end['Nb of Run'], fontsize=6, rotation=45)
        ax.set_ylim(np.min(summary_end[name])*0.9, np.max(summary_end[name]*1.1))
        pdf_end.savefig(fig)
        plt.close(fig)

    row = summary_min.shape[0]
    no_run = range(1, row+1)
    for ind, name in enumerate(prm_user['Names']):
        if name[0] in ['R', 'D', 'M']:
            unit = '/$\Omega$'
        elif name[0] in ['W']:
            unit = '/$\Omega \cdot s^{-1/2}$'
        elif name[0] in ['C']:
            unit = '/F'
        elif name[0] in ['L']:
            unit = '/H'
        elif name[0] in ['Q']:
            unit = '/$\Omega ^{-1} \cdot s^{n}$'
        elif name[0] in ['n', 'N']:
            unit = ''
        else:
            unit = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.ticklabel_format(scilimits=scilimits)
        ax.grid()
        ax.set_title('{0:s} vs Run'.format(name))
        ax.set_xlabel('No Run')
        ax.set_ylabel(r'Values {1:s}'.format(name, unit))
        ax.plot(no_run, summary_min[name], color='k', marker='o', mfc='w', mec='k', ls='-', lw=1, ms=4, mew=1)
        ax.set_xticks(no_run)
        ax.set_xticklabels(summary_min['Nb of Run'], fontsize=6, rotation=45)
        ax.set_ylim(np.min(summary_min[name])*0.9, np.max(summary_min[name]*1.1))
        pdf_min.savefig(fig)
        plt.close(fig)

    pdf_end.close()
    pdf_min.close()


def _random_scan(w, prm_array, immittance_exp_complex, symbolic_immittance, numeric_immittance, loops=1):
    # prm_array_random = _initialize_prm_from_immittance(symbolic_immittance)
    prm_array_random_min = _initialize_prm_from_immittance(symbolic_immittance)

    # 1st random scan
    prm_array_random = _get_random_prm_values(prm_array, all_parameters=True)
    immittance_calc_complex = numeric_immittance(prm_array_random['Values'], w)

    prm_array_random_min[:] = prm_array_random[:]
    distance = _get_distance(immittance_exp_complex, immittance_calc_complex)
    distance_min = distance

    for i in range(loops - 1):
        prm_array_random = _get_random_prm_values(prm_array, all_parameters=True)
        immittance_calc_complex = numeric_immittance(prm_array_random['Values'], w)
        distance = _get_distance(immittance_exp_complex, immittance_calc_complex)
        valid = _check_validity_prm(prm_array)

        if valid:
            if distance < distance_min:
                distance_min = distance
                prm_array_random_min[:] = prm_array_random[:]

    return prm_array_random_min


def _callback_fit(filename, run, nb_run, fit, nb_minimization,
                  distance, valid,
                  lcc_results, prm_array, prm_user, additional_messages=''):
    # progressbar_length = 10.0

    os.system('cls' if os.name == 'nt' else 'clear')

    sys.stdout.write(filename + '\n')
    sys.stdout.write('***** Run = {0:02d}/{1:02d} ***** \n'.format(run + 1, nb_run))
    sys.stdout.write('Minimizing ...\n')

    general_tb = ptb.PrettyTable(['Fit', 'log10(D)', 'Valid'])
    general_tb.add_row(['{0:03d}/{1:03d}'.format(fit+1, nb_minimization),
                        '{0:+09.4f}'.format(np.log10(distance)),
                        '{0:s}'.format(str(valid))])
    sys.stdout.write(general_tb.get_string() + '\n')

    lcc_tb = ptb.PrettyTable()
    lcc_tb.add_column('', ['Module', 'Phase', 'Re', 'Im'], align='l')
    lcc_tb.add_column('LCC', lcc_results[0:4], align='l')
    sys.stdout.write(lcc_tb.get_string() + '\n')

    prm = _update_prm(prm_array, prm_user)
    tb = ptb.PrettyTable()
    tb.add_column('Names', prm['Names'], align='l')
    tb.add_column('Values', prm['Values'], align='l')
    tb.add_column('Errors', prm['Errors'], align='l')
    tb.add_column('Fixed', prm['Fixed'], align='l')
    tb.add_column('Min', prm['Min'], align='l')
    tb.add_column('Max', prm['Max'], align='l')
    sys.stdout.write(tb.get_string() + '\n')
    for i in additional_messages:
        sys.stdout.write(i + '\n')
    sys.stdout.flush()


def _get_circuit_string(circuit):
    circuit_str = circuit.replace('+', '_s_').replace('/', '_p_')
    return circuit_str


def _save_fit_settings(circuit,
                       immittance_type,
                       datafilepath,
                       f_limits,
                       prmfilepath,
                       nb_processes,
                       nb_run_per_process,
                       nb_minimization,
                       init_type_0,
                       random_loops,
                       init_type_n,
                       init_type_validation,
                       xtol,
                       ftol,
                       maxiter_per_parameter,
                       maxfun_per_parameter,
                       fit_folder):
    datafilepath = os.path.abspath(datafilepath)
    prmfilepath = os.path.abspath(prmfilepath)
    datafilename, ext = os.path.basename(datafilepath).split('.')
    circuit_str = circuit.replace('+', '_s_').replace('/', '_p_')
    f_start, f_end = f_limits

    fit_settings = ['Circuit={0:s}'.format(circuit),
                    'Immittance Type={0:s}'.format(immittance_type),
                    'Experimental Data File={0:s}'.format(datafilepath),
                    'Frequency Range (Hz)={0:.2e},{1:.2e}'.format(f_start, f_end),
                    'Parameter File={0:s}'.format(prmfilepath),
                    'No of Processes={0:d}'.format(nb_processes),
                    'No of Runs={0:d}'.format(nb_run_per_process),
                    'No of Fits per Run={0:d}'.format(nb_minimization),
                    'Initialization Run 1={0:s}'.format(init_type_0),
                    'Random Loops={0:d}'.format(random_loops),
                    'Propagation Run > 1={0:s}'.format(init_type_n),
                    'Initialize after non valid parameters={0:s}'.format(init_type_validation),
                    'log10 xtol={0:.0f}'.format(np.log10(xtol)),
                    'log10 ftol={0:.0f}'.format(np.log10(ftol)),
                    'Iterations/Parameter={0:d}'.format(maxiter_per_parameter),
                    'fcalls/Parameter={0:d}'.format(maxfun_per_parameter),
                    'Result Folder={0:s}'.format(os.path.abspath(fit_folder))]

    text = '\n'.join(fit_settings)
    filepath = fit_folder + '/' + datafilename + '-' + circuit_str + '.' + FIT_SETTING_EXT
    filepath = os.path.abspath(filepath)
    fobj = open(filepath, 'w')
    fobj.write(text)
    fobj.close()


# noinspection PyPep8
def _create_fit_folder(root, datafilepath, alloy, alloy_id, circuit, init_types, f, f_limits, prm_user):
    datafilepath = os.path.abspath(datafilepath)
    datafilename, ext = os.path.basename(datafilepath).split('.')

    init_type_0, init_type_validation, init_type_n = init_types

    if f_limits is None:
        f_limits = (np.min(f), np.max(f))
    f_start, f_end = f_limits

    alloy = alloy.replace(' ', '_')
    alloy_id = alloy_id.replace(' ', '_')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')

    if root is None:
        root = './'
    if alloy is None:
        alloy = 'Unspecified_Alloy'
    if alloy_id is None:
        alloy_id = 'Unspecified_ID'

    if len(alloy) == 0:
        alloy = 'Unspecified_Alloy'
    if len(alloy_id) == 0:
        alloy_id = 'Unspecified_ID'

    alloy_folder = root + '/' + alloy + '-' + alloy_id
    circuit_str = _get_circuit_string(circuit)
    fit_folder = alloy_folder + '/' + timestamp + '-' + alloy + '-' + alloy_id + '-' + circuit_str + '-' + \
                 init_type_0[0].capitalize() + init_type_validation[0].capitalize() + init_type_n[0].capitalize() + \
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
    np.savetxt(filepath, prm_user, fmt=PRM_FORMATS_STR, delimiter='\t', newline='\n', header='\t'.join(PRM_NAMES))

    return fit_folder


def _initiliaze_prm_arrays(symbolic_immittance, prmfilepath):
    prmfilepath = os.path.abspath(prmfilepath)
    prm_user = _import_prm_file(prmfilepath)
    _check_parameter_names(prm_user, symbolic_immittance)
    _check_parameter_values(prm_user)

    prm_init = _initialize_prm_from_immittance(symbolic_immittance)
    prm_array = _initialize_prm_from_immittance(symbolic_immittance)
    prm_min_run = _initialize_prm_from_immittance(symbolic_immittance)
    prm_end_run = _initialize_prm_from_immittance(symbolic_immittance)

    prm_array = _update_prm(prm_user, prm_array)
    prm_init = _update_prm(prm_user, prm_init)
    prm_min_run = _update_prm(prm_user, prm_min_run)
    prm_end_run = _update_prm(prm_user, prm_end_run)

    return prm_user, prm_init, prm_array, prm_min_run, prm_end_run


def _initiliaze_minimization_array(nb_minimization, prm_user):
    header_minimization_results = '\t'.join(HEADER_MINIMIZATION_ELEMENTS) + '\t'
    header_minimization_results += '\t'.join(prm_user['Names'])
    col = len(HEADER_MINIMIZATION_ELEMENTS) + prm_user['Names'].size
    minimization_results = np.zeros(shape=(nb_minimization, col), dtype=FLOAT)

    return header_minimization_results, minimization_results


def import_experimental_data(filepath, immittance_type='Z'):
    r"""

    Import experimental data and compute the complex impedance.

    Supported files are .z files recorded by ZView software, .par files recorded by VersaStudio and .data files
    were the first three columns represent :math:`f`, :math:`ReZ`, :math:`ImZ`.

    Frequencies, real and imaginary parts of the impedance are extracted from the files.

    Parameters
    -----------
    filepath: string
        Path to the experimental data file.
    immittance_type: str, optional
        Type of immittance to be imported. It can be impedance or admittance i.e. Z or Y.

    Returns
    --------
    f: 1d numpy array
        Frequency vector.

    w: 1d numpy array
        Angular frequency computed as :math:`2 \pi f`

    I_exp_complex: 1d numpy array
        Complex electrochemical immittance computed as :math:`ReZ+jImZ`.

    """

    datafilepath = os.path.abspath(filepath)
    f, rez, imz = _get_exp_data(datafilepath)
    w = 2 * np.pi * f
    immittance_exp_complex = rez + 1j * imz

    if immittance_type == 'Y':
        immittance_exp_complex = 1.0/immittance_exp_complex

    return f, w, immittance_exp_complex


# noinspection PyTypeChecker
def generate_calculated_values(prmfilepath, savefilepath,
                               immittance_type='Z',
                               f_limits=(1e-3, 1e6),
                               points_per_decade=10,
                               re_relative_error=0.0, im_relative_error=0.0,
                               samples=100):
    r"""

    Generate values for a circuit from parameter values provided by the user.

    Parameters
    -----------
    prmfilepath: string
        Path to the parameter file.

    savefilepath: string
        Path the file where the data will be saved. If a file already exists, the content will be deleted.

    immittance_type: string
        Type of immittance to be used for generating the symbolic and numeric expression.
        Can be impedance or admittance i.e. Z or Y.

    f_limits: tuple of floats, optional
        Frequency range for the minimization procedure i.e. (lowest_frequency, highest_frequency).

    points_per_decade: int
        Number of points per decade for the frequency vector.

    Re_relative_error: float, optional
        Relative error for the real part for each frequency expressed in percentage.

    Im_relative_error: float, optional
        Relative error for the imaginary part for each frequency expressed in percentage.

    samples: int, optional
        Number of calculated values per frequency. It used to compute the confidence interval.

    """
    circuit = _get_circuit_from_prm(prmfilepath)

    # Symbolic Immittance
    immittance = cdp.get_symbolic_immittance(circuit, immittance_type=immittance_type, simplified=False)
    immittance_num = cdp.get_numeric_immittance(immittance)

    # check and import parameters
    prm_user, prm_init, prm_array, prm_min_run, prm_end_run = _initiliaze_prm_arrays(immittance, prmfilepath)

    f_start, f_end = f_limits
    logf_start, logf_end = np.log10(f_start), np.log10(f_end)
    decades = np.absolute(logf_end - logf_start)
    f = np.logspace(logf_start, logf_end, points_per_decade * decades)
    w = 2 * np.pi * f

    re_array = np.zeros(shape=(w.size, samples + 3), dtype=FLOAT)
    im_array = np.zeros(shape=(w.size, samples + 3), dtype=FLOAT)

    immittance_calc_complex = immittance_num(prm_array['Values'], w)
    mod, phase, re, im = _get_complex_parameters(immittance_calc_complex, deg=True)

    # The relative errors are taken as the 99% interval in normal distribution i.e. 3*sigma
    # sigma(w) = relative_error*Re(w)/3.0
    for i in range(samples):
        re_array[:, i] = re * (1 + np.random.standard_normal((w.size,)) * re_relative_error / 100.0 / 3.0)
        im_array[:, i] = im * (1 + np.random.standard_normal((w.size,)) * im_relative_error / 100.0 / 3.0)

    dof = samples - 1
    tvp = t.isf((1 - 0.99) / 2, dof)

    re_array[:, samples] = np.mean(re_array[:, 0:samples], axis=1)
    im_array[:, samples] = np.mean(im_array[:, 0:samples], axis=1)

    re_array[:, samples + 1] = np.std(re_array[:, 0:samples], axis=1, ddof=dof) / np.sqrt(samples) * tvp
    im_array[:, samples + 1] = np.std(im_array[:, 0:samples], axis=1, ddof=dof) / np.sqrt(samples) * tvp

    re_array[:, samples + 2] = np.absolute(re_array[:, samples + 1] / re_array[:, samples] * 100.0)
    im_array[:, samples + 2] = np.absolute(im_array[:, samples + 1] / im_array[:, samples] * 100.0)

    header_elements = [u'f /Hz',
                       u'Re{0:s} /Ohms'.format(immittance_type),
                       u'Im{0:s} /Ohms'.format(immittance_type),
                       u'd_Re{0:s} /Ohms'.format(immittance_type),
                       u'd_Im{0:s} /Ohms'.format(immittance_type),
                       u'd_Re{0:s} /%'.format(immittance_type),
                       u'd_Im{0:s} /%'.format(immittance_type)]

    re = re_array[:, samples]
    im = im_array[:, samples]

    d_re = re_array[:, samples + 1]
    d_im = im_array[:, samples + 1]

    drel_re = np.ceil(re_array[:, samples + 2])
    drel_im = np.ceil(im_array[:, samples + 2])

    data = np.vstack((f, re, im, d_re, d_im, drel_re, drel_im)).transpose()
    np.savetxt(savefilepath, X=data, delimiter='\t', header='\t'.join(header_elements))


def _round_errors(errors):
    r"""
    The errors are rounded by keeping one significant figure.

    ..seealso::  K. Protassov, Analyse statistique de données expérimentales, 1st ed. EDP Sciences, 2002.

    Parameters
    ----------
    errors: 1d numpy array
        Vector containing the errors to be rounded.

    Returns
    -------
    errors: 1d numpy array
        Vector containing the rounded errors.

    """
    log_errors = np.log10(errors)
    log_errors = np.floor(log_errors)
    errors = np.ceil(errors*10**(-log_errors))*10**log_errors
    return errors


# noinspection PyUnboundLocalVariable,PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
def run_fit(datafilepath, prmfilepath,
            nb_run_per_process=3, nb_minimization=50,
            f_limits=None, init_types=('random', 'random', 'random'), immittance_type='Z',
            root=None, alloy=None, alloy_id=None,
            random_loops=DEFAULT_RANDOM_LOOPS, nb_processes=1, process_id=1, simplified=False,
            maxiter_per_parameter=DEFAULT_MAXITER_PER_PARAMETER,
            maxfun_per_parameter=DEFAULT_MAXFUN_PER_PARAMETER,
            xtol=DEFAULT_XTOL, ftol=DEFAULT_FTOL,
            full_output=True, retall=False, disp=False, fmin_callback=None, callback=None, display=False):
    r"""

    `nb_run_per_process` will be started with the `init_types` initialization. `nb_minimization` will be performed by
    calling the `fmin` procedure (see Scipy documentation for more details).
    After each call of the `fmin` procedure, the parameter values are check if there are restricted to the boundaries
    fixed by the user in the parameter file.

    The parameter corresponding to the minimal distance and the final distance for each run are saved in text files.
    All results of each minimization are also stored in text file.
    The corresponding Nyquist, Bode Phase and Modulus plots are plotted for the minimal and the final distance.

    A folder is created in the `root` folder which name is created after the `alloy` and `alloy_id` options.
    The naming of the folder follows the following template:
    `%Y-%m-%d-%H%M%S-alloy-alloy_id-circuit-init_types-start_frequency_Hz-end_frequency_Hz`.

    The results for each run are finally summarized by listing the result files for parameters
    at the end and the minimum of each run.
    Compute the distance, the LCCs for the frequency range that was used for minimizing the target function.
    The results are saved in 2 files: .SumEnd, .SumMin.
    The results are plotted and saved in 2 files: -0-End.pdf, -0-Min.pdf.


    Parameters
    -----------
    datafilepath: string
        Path to the data file.

    prmfilepath: string
        Path to the parameter file.

    nb_run_per_process: int, optional
        Number of run per started process.

    nb_minimization: int, optional
        Number of minimization through the fmin procedure (see Scipy documentation for more details).

    init_types: tuple of string, optional
        Initialization types:

        * for the first run: available options are random, user
        * after invalid parameters in minimization procedure: available options are random, user
        * for the following run i.e. run > 1: available options are random, user, min, end

    f_limits: tuple of floats, optional
        Frequency range for the minimization procedure i.e. (lowest_frequency, highest_frequency).
        If f_limits is None the limits are set to the lowest and highest experimental frequency.

    immittance_type: string
        Type of immittance to be used for generating the symbolic and numeric expression.
        Can be impedance or admittance i.e. Z or Y.

    root: string
        Path of the folder were the results will be saved. If `root` is set to None, the current working
        directory is used.

    alloy: string
        Alloy identification. If `alloy` is set to None or is an empty string, `alloy` is set to
        `Unspecified_Alloy`.

    alloy_id: string
        Sample identification. If `alloy_id` is set to None or is an empty string, `alloy_id` is set to
        `Unspecified_Alloy_ID`.

    random_loops: int, optional
        Number of random loops to be performed in the case of random initialization.

    nb_processes: int, optional
        Number of processes to be started.

    process_id: int, optional
        Running process identification.
        
    simplified: bool, optional
        Flag indicating if the symbolic expression of the immittance must be simplified or not.

    maxiter_per_parameter : int, optional
        Maximum number of iterations to perform.

    maxfun_per_parameter : number, optional
        Maximum number of function evaluations to make.

    xtol : float, optional
        Relative error in xopt acceptable for convergence.
        
    ftol : float, optional
        Relative error in func(xopt) acceptable for convergence.

    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.

    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        
    disp : bool, optional
        Set to True to print convergence messages.

    fmin_callback : callable, optional
        Called after each iteration in the fmin procedure, as callback(xk), where xk is the
        current parameter vector. See Scipy documentation.

    callback: callable, optional
        Called after each minimization, as
        callback(filename, run, nb_run_per_process, fit, nb_minimization, distance, valid, LCC_results,
        prm_end_run, prm_user).

    display: bool, optional
        If set to True, the output for each minimization is displayed. A display function is passed to the callback.
        Set it to False if you want to pass your own callback.
        
    Returns
    --------
    None


    """
    if display:
        callback = _callback_fit
    circuit = _get_circuit_from_prm(prmfilepath)

    # Symbolic Immittance
    immittance = cdp.get_symbolic_immittance(circuit, immittance_type=immittance_type, simplified=simplified)
    immittance_num = cdp.get_numeric_immittance(immittance)

    # check and import parameters
    prm_user, prm_init, prm_array, prm_min_run, prm_end_run = _initiliaze_prm_arrays(immittance, prmfilepath)
    mask_to_fit = _get_mask_to_fit(prm_array)
    nb_param = mask_to_fit.size

    # import data
    datafilepath = os.path.abspath(datafilepath)
    filename = os.path.basename(datafilepath)
    f, w, immittance_exp_complex = import_experimental_data(datafilepath, immittance_type=immittance_type)
    mask = _get_frequency_mask(f, f_limits)

    # set the initialization types
    init_type_0, init_type_validation, init_type_n = init_types

    # create fit folder
    fit_folder = _create_fit_folder(root, datafilepath, alloy, alloy_id, circuit, init_types, f, f_limits, prm_user)
    circuit_str = _get_circuit_string(circuit)

    # initiliaze the minimization array for storing results from minimization loops
    header_minimization_results, minimization_results = _initiliaze_minimization_array(nb_minimization, prm_user)

    # save fit settings
    _save_fit_settings(circuit,
                       immittance_type,
                       datafilepath,
                       f_limits,
                       prmfilepath,
                       nb_processes,
                       nb_run_per_process,
                       nb_minimization,
                       init_type_0,
                       random_loops,
                       init_type_n,
                       init_type_validation,
                       xtol,
                       ftol,
                       maxiter_per_parameter,
                       maxfun_per_parameter,
                       fit_folder)

    for run in range(nb_run_per_process):

        if run == 0:
            if init_type_0 == 'random':
                prm_array = _random_scan(w, prm_array, immittance_exp_complex, immittance, immittance_num,
                                         loops=random_loops)
            elif init_type_0 == 'user':
                prm_array[:] = prm_init[:]
        else:
            if init_type_n == 'random':
                prm_array = _random_scan(w, prm_array, immittance_exp_complex, immittance, immittance_num,
                                         loops=random_loops)
            elif init_type_n == 'user':
                prm_array[:] = prm_init[:]
            elif init_type_n == 'min':
                prm_array[:] = prm_min_run[:]
            elif init_type_n == 'end':
                prm_array[:] = prm_end_run[:]

        immittance_calc_complex = immittance_num(prm_array['Values'], w)
        distance = _get_distance(immittance_exp_complex, immittance_calc_complex)
        distance_min_run = distance
        distance_end_run = distance

        for fit in range(nb_minimization):
            prm_array, distance = _minimize(w=w[mask], immittance_exp_complex=immittance_exp_complex[mask],
                                            immittance_num=immittance_num,
                                            prm_array=prm_array,
                                            maxiter=maxiter_per_parameter*nb_param,
                                            maxfun=maxfun_per_parameter*nb_param,
                                            xtol=xtol, ftol=ftol,
                                            full_output=full_output, retall=retall, disp=disp, callback=fmin_callback)

            prm_output = _update_prm(prm_array, prm_user)
            immittance_calc_complex = immittance_num(prm_array['Values'], w)
            lcc_results = _get_lcc(immittance_exp_complex[mask], immittance_calc_complex[mask])
            valid = _check_validity_prm(prm_array)
            minimization_results[fit] = np.hstack((fit + 1,
                                                   int(valid),
                                                   np.log10(distance),
                                                   lcc_results,
                                                   prm_output['Values']))

            prm_end_run[:] = prm_array[:]
            distance_end_run = distance

            args = (filename, run, nb_run_per_process, fit, nb_minimization, distance, valid,
                    lcc_results, prm_array, prm_user)
            if callback is not None:
                callback(*args)

            if not valid:
                if init_type_validation == 'random':
                    prm_array = _get_random_prm_values(prm_array, all_parameters=False)
                elif init_type_validation == 'user':
                    prm_array[:] = prm_init[:]

            elif valid:
                if distance_min_run > distance:
                    distance_min_run = distance
                    prm_min_run[:] = prm_array[:]

        if callback is not None:
            callback(filename, run, nb_run_per_process, fit, nb_minimization,
                     distance, valid,
                     lcc_results, prm_end_run, prm_user, ['Computing Covariance Matrix...'])

        prm_end_run['Errors'][:] = _get_prm_error(prm_end_run['Values'],
                                                  _get_residuals,
                                                  _EPSILON,
                                                  w[mask],
                                                  immittance_exp_complex[mask],
                                                  immittance_num)
        prm_min_run['Errors'][:] = _get_prm_error(prm_min_run['Values'],
                                                  _get_residuals,
                                                  _EPSILON,
                                                  w[mask],
                                                  immittance_exp_complex[mask],
                                                  immittance_num)

        if callback is not None:
            callback(filename, run, nb_run_per_process, fit, nb_minimization,
                     distance, valid,
                     lcc_results, prm_end_run, prm_user, ['Saving Results...'])

        _save_results(circuit, run, process_id, fit_folder, datafilepath, circuit_str, f, mask,
                      immittance_exp_complex, immittance_num,
                      prm_user, prm_min_run, prm_end_run, distance_min_run, distance_end_run,
                      minimization_results, header_minimization_results)

    if callback is not None:
        callback(filename, run, nb_run_per_process, fit, nb_minimization,
                 distance, valid,
                 lcc_results, prm_end_run, prm_user,
                 additional_messages=['Computing Summary ...'])

    _get_summary(fit_folder, immittance, immittance_num)
    _plot_summary(fit_folder)


# noinspection PyUnresolvedReferences
def _get_prm_error(p, func, epsilon, *args):
    r"""
    Compute the errors of the parameters with interval confidence of 0.95 by estimating numerically the covariance
    matrix using the Jacobian matrix.

    Parameters
    -----------
    p: 1d numpy array
        Vector containing the parameters values.
    func: function
        Function to be used for the numerical estimation of the Jacobian matrix
    epsilon: float
        Step to be used in the numerical estimation of the Jacobian.
    args: tuple
        Arguments to be passed to `func`.
    """

    n = args[0].size
    nb_param = p.size
    dof = n-nb_param-1
    tvp = t.isf(0.05/2.0, dof)
    if dof <= 0:
        raise np.linalg.LinAlgError('Degree of freedom is lower or equal to zero. Too many parameters are fitted.')
    try:
        jac = approx_jacobian(p, func, epsilon, *args)
        cov = np.dual.inv(np.dot(jac.T, jac))
        g = _get_chi2(p, *args)/dof
        dp = _round_errors(np.sqrt(cov.diagonal()*g)*tvp)
    except np.linalg.LinAlgError as error:
        print(error.message)
        dp = np.ones(shape=p.shape, dtype=FLOAT)*-1

    return dp