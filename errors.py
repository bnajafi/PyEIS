# -*- coding: utf-8 -*-

r"""
Documentation
"""


class FileTypeError(Exception):
    """

    Raise an error when data file type was not recognized.

    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(FileTypeError, self).__init__(message, *args)


class ParameterNameError(Exception):
    """ Raise an error when the names of the parameter given by the
    user do not correspond to the names of the parameters detected
    in the symbolic expression of the immittance.
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(ParameterNameError, self).__init__(message, *args)


class ParameterNumberError(Exception):
    """ Raise an error when the number of parameter given by the
    user do not correspond to the number of parameters detected
    in the symbolic expression of the immittance.
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(ParameterNumberError, self).__init__(message, *args)


class ParameterValueError(Exception):
    """

    Raise an error when the parameter values are set to zero.

    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(ParameterValueError, self).__init__(message, *args)


class UnknownComponentError(Exception):
    """
    Raise an error when an unknown component is detected.
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(UnknownComponentError, self).__init__(message, *args)


class DoubleSignError(Exception):
    """
    Raise an error when double signs i.e. ++, +/, //, /+ are detected.
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(DoubleSignError, self).__init__(message, *args)


class BracketMismacthError(Exception):
    """
    Raise an error when the number of opening and closing brackets are not equal.
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(BracketMismacthError, self).__init__(message, *args)


class ImmittanceTypeError(Exception):

    """
    Raise an error when the immittance type is not recognized i.e when the
    immittance is not impedance (Z) or admittance (Y).
    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(ImmittanceTypeError, self).__init__(message, *args)


class ConnexionTypeError(Exception):
    """

    Raise an error when the connexion type is not recognized i.e when the connexion is not serie or parallel.

    """

    def __init__(self, message, *args):
        self.message = message

        # allow users to initialize misc. arguments as any other builtin errors
        super(ConnexionTypeError, self).__init__(message, *args)   
