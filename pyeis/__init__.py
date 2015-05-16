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
 * Numpy >1.8
 * Scipy >0.14
 * Sympy >0.7.5
 * Matplotlib >1.2

Generating EIS data for a given circuit
---------------------------------------

Let's compute the impedance of a simple electrical circuit: `Rel+Rct/Qdl`. The computation function require a parameter file (the extension does not matter) which contains the information for each electrical component.

.. code-block:: python

  import eis_functions

  # set the needed parameter in order to
  # calculate the EIS data
  circuit = 'Rel+Rct/Qdl'
  prmfilepath = './data/test.PrmInit'
  savefilepath = './data/Generated_Data.data'
  f_limits = (1e-3, 1e9)

  eis_functions.generate_calculated_values(circuit=circuit,
                                           prmfilepath=prmfilepath,
                                           savefilepath=savefilepath,
                                           f_limits=f_limits,
                                           points_per_decade=10,
                                           re_relative_error=5.0,
                                           im_relative_error=5.0,
                                           samples=3)
Let's visualize the generated data

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt

  #Visualize the computed data
  f, Re, Im = np.loadtxt(savefilepath, usecols=(0, 1, 2), delimiter='\t', unpack=True)
  plt.figure()
  plt.plot(Re, Im, marker='o')
  plt.gca().set_aspect('equal')
  ymin, ymax = plt.ylim()
  plt.ylim(ymax, ymin)
  plt.show()

Fitting experimental data
-------------------------

.. code-block:: python

  # Fit the generated file
  datafilepath = savefilepath

  # 3 parameters are mandatory i.e. circuit, datafile, and parameter file.
  # Other parameters have default values and are not required in order to run the fitting procedure

  # noinspection PyProtectedMember
  fit_options = {'nb_run_per_process': 3,
                 'nb_minimization': 50,
                 'init_types': ('random', 'random', 'random'),
                 'f_limits': f_limits,
                 'immittance_type': 'Z',
                 'root': './data/fit_results/',
                 'alloy': 'test',
                 'alloy_id': '1',
                 'random_loops': 200,
                 'process_id': 1,
                 'simplified': False,
                 'callback': eis_functions._callback_fit,
                 'maxiter_per_parameter': 200,
                 'maxfun_per_parameter': 200,
                 'xtol': 1e-30,
                 'ftol': 1e-30,
                 'full_output': True,
                 'retall': False,
                 'disp': False,
                 'fmin_callback': None}

  eis_functions.run_fit(circuit, datafilepath, prmfilepath, **fit_options)

License information
-------------------

See the file ``LICENSE`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
"""
from __future__ import absolute_import

from . import eis_functions
from . import circuit_decomposition
from . import version

__version__ = version

