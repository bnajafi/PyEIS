"""
PyEIS
=====

Contains core functions for computing and fitting experimental data from Electrochemical Impedance Spectroscopy.

The functions are primarily aimed to be used in python scripts in order to automate the computation of EIS data or
fitting experimental data.

The documentation is 80% completed and the code is still under active development. However, great care is taken
in avoiding backward incompatibilities.

How to install
--------------
Download the zip or tarball file and extract it locally. Install the package by using the setup.py file.

.. code-block:: python

    python setup.py install

Numpy, Scipy, Sympy and Matplotlib are required dependencies:
 * Numpy >=1.8
 * Scipy >=0.14
 * Sympy >=0.7.5
 * Matplotlib >=1.2
 * PrettyTables >=0.7.2

License information
-------------------

See the file ``LICENSE`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
"""

from __future__ import absolute_import

from . import eis_functions as eis
from . import circuit_decomposition as cdp
from . import version

__version__ = version.version

