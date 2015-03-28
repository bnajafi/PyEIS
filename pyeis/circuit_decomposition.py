# -*- coding: utf-8 -*-

"""
This module contains all necessary functions for determining and instantiating the symbolic and numeric expressions of
the immittance based on a electrical circuit provided as a string. The circuit is parsed in order to detect the
electrical component and the different connexions i.e. series or parallel.


"""
import sys
import sympy as sym
from pyeis.errors import UnknownComponentError, DoubleSignError, BracketMismacthError
from pyeis.errors import ImmittanceTypeError, ConnexionTypeError


# Shadowing the built-in zip from Python 2.7
# Compatibility with Python 3
if sys.version_info[0] == 2:
    # noinspection PyShadowingBuiltins
    from itertools import izip as zip

# CONSTANTS
w = sym.Symbol('w', real=True)
p = sym.Symbol('p', real=True)
R = sym.Symbol('R', real=True)
C = sym.Symbol('C', real=True)
L = sym.Symbol('L', real=True)
Q = sym.Symbol('Q', real=True)
n = sym.Symbol('n', real=True)
W = sym.Symbol('W', real=True)
Nw = sym.Symbol('Nw', real=True)
D = sym.Symbol('D', real=True)
Td = sym.Symbol('Td', real=True)
Nd = sym.Symbol('Nd', real=True)
M = sym.Symbol('M', real=True)
Tm = sym.Symbol('Tm', real=True)
Nm = sym.Symbol('Nm', real=True)

VARIABLE = w
PARAMETERS = p
SUBCIRCUIT_NAME = 'Z_i'
IMMITTANCE_TYPES = ['Z', 'Y']
CONNEXION_TYPES = ['series', 'parallel']
ELECTROCHEMICAL_COMPONENTS = {'R': {'Z': R,
                                    'Y': 1.0 / R},
                              'C': {'Z': -sym.I / (C * w),
                                    'Y': sym.I * C * w},
                              'L': {'Z': sym.I * L * w,
                                    'Y': -sym.I / (L * w)},
                              'Q': {'Z': 1.0 / (Q * w ** n * sym.I ** n),
                                    'Y': Q * w ** n * sym.I ** n},
                              'W': {'Z': W * (1 - sym.I) / w ** Nw,
                                    'Y': w ** Nw / (W * (1 - sym.I))},
                              'D': {'Z': D / ((Td * sym.I * w) ** Nd) * sym.tanh((Td * sym.I * w) ** Nd),
                                    'Y': 1.0 / D * ((Td * sym.I * w) ** Nd) * (1.0/sym.tanh((Td * sym.I * w) ** Nd))},
                              'M': {'Z': M / ((Tm * sym.I * w) ** Nm) * (1.0/sym.tanh((Tm * sym.I * w) ** Nm)),
                                    'Y': 1.0 / M * ((Tm * sym.I * w) ** Nm) * sym.tanh((Tm * sym.I * w) ** Nm)},
                              }

if SUBCIRCUIT_NAME[0] in ELECTROCHEMICAL_COMPONENTS.keys():
    SUBCIRCUIT_NAME[0] = 'E'

ALLOWED_CHARACTERS = ('R', 'C', 'Q', 'L', 'W', 'D', 'M',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      '(', ')', '+', '/', '_',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')


def _remove_space(circuit):

    r"""
    The function removes the extra spaces in the electrical circuit.

    Parameters
    ----------

    circuit : str
        Electrical circuit.

    Returns
    -------
    circuit : str
        Electrical circuit with no extra spaces.
    """

    circuit = circuit.replace(' ', '')

    return circuit


def _remove_unallowed_characters(circuit):
    r"""
    The function removes the unallowed characters of the electrical circuit and replaces them with '_'.

    Parameters
    ----------

    circuit : str
        Electrical circuit.

    Returns
    -------
    circuit : str
        Electrical circuit with no unallowed characters.
    """

    for char in circuit:
        if char not in ALLOWED_CHARACTERS:
            circuit = circuit.replace(char, '_')

    return circuit


def _check_unknown_components(circuit):
    r"""

    The function checks if the circuit contains unknown elements i.e.
    the elements that do not start with R, C, L, Q, W, D or M will be
    interpreted as unknown elements.

    Parameters
    ----------
    circuit : str
        Electrical circuit.

    Returns
    -------
    flag: bool
        True if unknown elements were detected

    unknown_elements: list
        List of the unknown elements

    """
    circuit = circuit.replace('(', '').replace(')', '')
    elements = []
    unknown_elements = []

    for i_series in circuit.split('+'):
        for i_parallel in i_series.split('/'):
            elements.append(i_parallel)

    for element in elements:
        component = element[0]
        if component not in ELECTROCHEMICAL_COMPONENTS.keys():
            unknown_elements.append(component)

    if len(unknown_elements) == 0:
        flag = False
    else:
        flag = True

    return flag, unknown_elements


def _check_double_signs(circuit):
    r"""
    The function checks if the circuit contains two consecutive '+' or '/' operators.

    Parameters
    ----------
    circuit : str
        Electrical circuit.

    Returns
    -------
    True or False : bool
    """
    flag = False
    n = len(circuit)
    for k in range(0, n - 1):
        if (circuit[k] == '+' or circuit[k] == '/') and (circuit[k + 1] == '+' or circuit[k + 1] == '/'):
            flag = True

    return flag


def _get_bracket_positions(circuit):
    r"""
    Find the positions of the opening and closing brackets.

    Parameters
    -----------
    circuit: string
        Electrical circuit.
    """
    brackets = []

    for ind, i in enumerate(circuit):

        if i == '(':
            brackets.append(['O', ind])
        elif i == ')':
            brackets.append(['C', ind])

    return brackets


def _check_brackets(circuit):

    r"""
    Check if brackets are present in the string representation of the electrical circuit. 

    Parameters
    -----------
    circuit: string
       Electrical circuit. 
    """

    mismatch_flag = True
    bracket_flag = True
    opening = 0
    closing = 0

    brackets = _get_bracket_positions(circuit)

    if len(brackets) == 0:
        bracket_flag = False
        mismatch_flag = False
    else:
        for i in brackets:
            if i[0] == 'O':
                opening += 1
            elif i[0] == 'C':
                closing += 1

        if opening == closing:
            mismatch_flag = False

    return bracket_flag, mismatch_flag, opening, closing


def _impedance_series(impedances):

    r"""
    Get the equivalent impedance for in series impedances.

    Parameters
    -----------
    impedances: list 
        List of sympy expressions representing the impedances.

    Returns
    -------
    equivalent_impedance: sympy expression
        Equivalent impedance.
    """

    equivalent_impedance = 0
    for z in impedances:
        equivalent_impedance += z

    return equivalent_impedance


def _impedance_parallel(impedances):

    r"""
    Get the equivalent impedance for in parallel impedances.

    Parameters
    -----------
    impedances: list 
        List of sympy expressions representing the impedances.

    Returns
    -------
    equivalent_impedance: sympy expression
        Equivalent impedance.
    """

    equivalent_impedance = 0
    for z in impedances:
        equivalent_impedance += 1.0 / z

    return 1.0 / equivalent_impedance


def _admittance_series(admittances):

    r"""
    Get the equivalent admittance for in series impedances.

    Parameters
    -----------
    impedances: list 
        List of sympy expressions representing the admittances.

    Returns
    -------
    equivalent_impedance: sympy expression
        Equivalent admittance.
    """

    equivalent_admittance = 0
    for y in admittances:
        equivalent_admittance += 1.0 / y

    return 1.0 / equivalent_admittance


def _admittance_parallel(admittances):

    r"""
    Get the equivalent admittance for in parallel admittances 

    Parameters
    ----------
    admittances: list
        List of sympy expressions representing the admittances.

    Returns
    -------
    equivalent_impedance: sympy expression
        Equivalent admittance.
    """

    equivalent_admittance = 0
    for y in admittances:
        equivalent_admittance += y

    return equivalent_admittance


def _get_equivalent_immittance(immittances, connexion="series", immittance_type="Z", simplified=False):

    r"""
    DocString
    """

    equivalent_immittance = 0

    if connexion not in CONNEXION_TYPES:
        message = "Unknown connexion type. Connexion types are {0:s}".format(str(IMMITTANCE_TYPES))
        raise ConnexionTypeError(message)

    if connexion == 'series':
        if immittance_type == 'Z':
            equivalent_immittance = _impedance_series(immittances)
        elif immittance_type == 'Y':
            equivalent_immittance = _admittance_series(immittances)

    elif connexion == "parallel":
        if immittance_type == 'Z':
            equivalent_immittance = _impedance_parallel(immittances)
        elif immittance_type == 'Y':
            equivalent_immittance = _admittance_parallel(immittances)

        if simplified:
            equivalent_immittance = sym.simplify(equivalent_immittance)

    return equivalent_immittance


def _get_element_immittance(element, subcircuits=list(), immittance_type="Z"):

    r"""
    DocString
    """

    substitution_flag = False

    component = element[0]
    suffix = element[1:]

    if component in ELECTROCHEMICAL_COMPONENTS.keys():
        element_immittance = ELECTROCHEMICAL_COMPONENTS[component][immittance_type]
        substitution_flag = True

    elif len(subcircuits) != 0:
        element = element.replace(SUBCIRCUIT_NAME, 'subcircuits')
        element_immittance = eval(element)

    else:
        message = "Unknown element: {0:s}".format(element)
        raise UnknownComponentError(message)

    if substitution_flag:
        parameters, parameter_names = _get_parameters(element_immittance)
        new_parameters = []
        for name in parameter_names:
            new_name = name + suffix
            new_parameters.append(sym.Symbol(new_name, real=True))

        subs_dic = dict(zip(parameters, new_parameters))
        element_immittance = element_immittance.subs(subs_dic)

    return element_immittance


def _get_parameters(immittance):

    r"""
    DocString
    """

    parameters = []
    parameter_names = []
    list_of_symbols = list(immittance.atoms())

    for i in list_of_symbols:
        if isinstance(i, sym.Symbol):
            if i.name[0] not in ['w', 'I']:
                parameters.append(i)
                parameter_names.append(i.name)

    return parameters, parameter_names


def _elementary_circuit_immittance(circuit, subcircuits=list(), immittance_type="Z", simplified=False):
    
    r"""
    DocString
    """
    series_elements = circuit.split('+')
    series_element_immittances = []

    if immittance_type in IMMITTANCE_TYPES:

        for series_element in series_elements:

            if '/' in series_element:

                parallel_elements = series_element.split('/')
                parallel_element_immittances = []

                for parallel_element in parallel_elements:
                    i_parallel_element = _get_element_immittance(parallel_element,
                                                                 subcircuits=subcircuits,
                                                                 immittance_type=immittance_type)
                    parallel_element_immittances.append(i_parallel_element)

                ieq_parallel = _get_equivalent_immittance(parallel_element_immittances,
                                                          connexion='parallel',
                                                          immittance_type=immittance_type,
                                                          simplified=simplified)
                series_element_immittances.append(ieq_parallel)

            else:
                i_series_element = _get_element_immittance(series_element,
                                                           subcircuits=subcircuits,
                                                           immittance_type=immittance_type)
                series_element_immittances.append(i_series_element)

        elementary_immittance = _get_equivalent_immittance(series_element_immittances,
                                                           connexion='series',
                                                           immittance_type=immittance_type,
                                                           simplified=simplified)

    else:
        message = "Unknown immittance type. Immittance types are {0:s}".format(str(IMMITTANCE_TYPES))
        raise ImmittanceTypeError(message)

    return elementary_immittance


def _get_immittance_in_brackets(circuit, immittance_type='Z', simplified=False):

    r"""
    DocString
    """

    subcircuits = []
    subcircuit_immittances = []

    brackets = _get_bracket_positions(circuit)
    n = len(brackets)
    subcircuit_counter = 0

    while n > 0:
        for i in range(n - 1):
            if (brackets[i][0] == 'O') and (brackets[i + 1][0] == 'C'):
                break
        start, end = brackets[i][1] + 1, brackets[i + 1][1]

        subcircuit = circuit[start:end]
        subcircuits.append(subcircuit)

        subcircuit_immittance = _elementary_circuit_immittance(subcircuit,
                                                               subcircuits=subcircuit_immittances,
                                                               immittance_type=immittance_type,
                                                               simplified=simplified)

        subcircuit_immittances.append(subcircuit_immittance)

        subcircuit_with_brackets = circuit[start - 1: end + 1]
        circuit = circuit.replace(subcircuit_with_brackets, '{0:s}[{1:d}]'.format(SUBCIRCUIT_NAME,
                                                                                  subcircuit_counter))

        brackets = _get_bracket_positions(circuit)
        n = len(brackets)
        subcircuit_counter += 1

    return circuit, subcircuits, subcircuit_immittances


def get_symbolic_immittance(circuit, immittance_type='Z',
                            simplified=False, elementary_circuit_output=False):
    r"""
    DocString
    """

    symbolic_immittance = 0

    circuit = _remove_space(circuit)

    flag_double_sign = _check_double_signs(circuit)

    if flag_double_sign:
        warning = "Doubled signs in the circuit."
        raise DoubleSignError(warning)

    else:
        flag_unknown_elements, unknown_elements = _check_unknown_components(circuit)

        if flag_unknown_elements:
            warning = "Unknown elements were detected: {0:s}.".format(str(unknown_elements))
            raise UnknownComponentError(warning)

        else:
            circuit = _remove_unallowed_characters(circuit)
            bracket_flag, mismatch_flag, opening, closing = _check_brackets(circuit)

            if bracket_flag is True and mismatch_flag is True:
                warning = "Number of opening and closing brackets are not equal."
                raise BracketMismacthError(warning)

            elif bracket_flag is True and mismatch_flag is False:

                results = _get_immittance_in_brackets(circuit,
                                                      immittance_type=immittance_type,
                                                      simplified=simplified)

                elementary_circuit, subcircuits, subcircuit_immittances = results

                if elementary_circuit_output:
                    print elementary_circuit
                symbolic_immittance = _elementary_circuit_immittance(elementary_circuit,
                                                                     subcircuits=subcircuit_immittances,
                                                                     immittance_type=immittance_type,
                                                                     simplified=simplified)
            elif bracket_flag is False:
                symbolic_immittance = _elementary_circuit_immittance(circuit,
                                                                     subcircuits=[],
                                                                     immittance_type=immittance_type,
                                                                     simplified=simplified)

    return symbolic_immittance


def get_numeric_immittance(symbolic_immittance):
    r"""
    DocString
    """

    parameters, parameter_names = _get_parameters(symbolic_immittance)
    nb_parameters = len(parameters)
    p_list = sym.symbols('p[0:{0:d}]'.format(nb_parameters), real=True)

    substitution_dict = {}
    for parameter, p_i in zip(parameters, p_list):
        substitution_dict[parameter] = p_i

    p_symbolic_immittance = symbolic_immittance.subs(substitution_dict, locals={})

    args = (PARAMETERS, VARIABLE)
    numeric_immittance = sym.lambdify(args, p_symbolic_immittance, modules="numpy", dummify=False)

    return numeric_immittance


if __name__ == '__main__':
    circuit = 'R@el+(Cdl+((Rf+C)/Qdl))+(R/Q)'
    print('##### Test get_symbolic_immittance ####')
    print('Input Circuit: {0:s}'.format(circuit))
    I = get_symbolic_immittance(circuit, immittance_type='Z', simplified=False)
    sym.pprint(I)

    I_num = get_numeric_immittance(I)
    print('Numeric Immittance:')
    print(I_num)
