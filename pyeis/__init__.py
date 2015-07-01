"""
Generate symbolic and lambda functions from the string representation of an electronic circuit in order to compute EIS data.
"""

from __future__ import absolute_import

from . import eis_functions as eis
from . import circuit_decomposition as cdp
from . import version

__version__ = version.version

