PyEIS
======

Contains core functions for computing and fitting experimental data from Electrochemical Impedance Spectroscopy. 

The functions are primarily aimed to be used in python scripts in order to automate the generation of calculated EIS data or fitting of experimental EIS data.

The documentation is still incomplete and the code is still under active development.

How to install
---------------
At this point, PyEIS cannot be installed as a standard python package. However, the core functions found in **circuit_decompostion.py** and **eis_functions** are usuable. Most errors are correctly handled and the documentation is 75% completed.

Download the zip or tarball file and extract it locally. In your python script, add the path to the PyEIS folder in order to be able to import modules.

Numpy, Scipy, Sympy and Matplotlib are required dependencies:
 * Numpy >1.8
 * Scipy >0.14
 * Sympy >0.7.5
 * Matplotlib >1.2

.. code-block:: python

  import sys
  sys.path.append('/path/to/the/PyEIS/folder/')

Generating EIS data for a given circuit
----------------------------------------

Let's compute the impedance of a simple electrical circuit: `Rel+Rct/Qdl`.

.. code-block:: python

  import eis_functions
  
  #set the needed parameter in order to
  #calculate the EIS data
  circuit = 'Rel+Rct/Qdl'
  prmfilepath = './data/test.PrmInit'
  savefilepath = './data/Generated_Data.data'
  f_limits=(1e-3, 1e9)

  eis_functions.generate_calculated_values(circuit=circuit,
                                         prmfilepath=prmfilepath,
                                         savefilepath=savefilepath,
                                         f_limits=f_limits,
                                         points_per_decade=10,
                                         sigma=5.0)

Let's visualize the generated data

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  #Visualize the computed data
  f, Re, Im = np.loadtxt(savefilepath, usecols=(0,1,2), delimiter='\t', unpack=True)

  plt.plot(Re, Im, marker='o')
  plt.gca().set_aspect('equal')
  ymin, ymax = plt.ylim()
  plt.ylim(ymax, ymin)

  plt.show()

Fitting experimental data
--------------------------

.. code-block:: python

  #fit the data generated in the previous section
  datafilepath = savefilepath
  eis_functions.run_fit(circuit, datafilepath, prmfilepath,
        nb_run_per_process=20, nb_minimization=50,
        init_types=('random','random','random'), f_limits=f_limits, immittance_type = 'Z',
        root = './data/fit_results/', alloy='test', alloy_id='1',
        random_loops=200, process_id=1, simplified=False,
        maxiter_per_parameter=200, maxfun_per_parameter=200, xtol=1e-30, ftol=1e-30,
        full_output=True, retall=False, disp=False, fmin_callback=None, callback=eis_functions._callback_fit)

License information
-------------------

See the file ``LICENSE`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
