import matplotlib.pyplot as plt
import numpy as np

import pyeis.circuit_decomposition as cdp
import pyeis.eis_functions as eis_func


# Decomposition of the circuit
# Symbolic and numeric versions can be retrieved
# Symbolic and numeric expressions can be used for further calculations
circuit = 'Dox'
I = cdp.get_symbolic_immittance(circuit, immittance_type='Z', simplified=False)
print(I)
I_num = cdp.get_numeric_immittance(I)
print('Numeric Immittance:')
print(I_num)

# generate data for a given circuit
# calls the two previously mentioned functions
prmfilepath = './data/test.PrmInit'
savefilepath = './data/Generated_Data.data'
f_limits = (50e-6, 1e6)

eis_func.generate_calculated_values(prmfilepath=prmfilepath,
                                    savefilepath=savefilepath,
                                    f_limits=f_limits,
                                    points_per_decade=17,
                                    re_relative_error=1.0,
                                    im_relative_error=1.0,
                                    samples=3)

f, Re, Im = np.loadtxt(savefilepath, usecols=(0, 1, 2), delimiter='\t', unpack=True)
plt.figure()
plt.plot(Re, Im, marker='o')
plt.gca().set_aspect('equal')
ymin, ymax = plt.ylim()
plt.ylim(ymax, ymin)

plt.show()


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
               'callback': eis_func._callback_fit,
               'maxiter_per_parameter': 200,
               'maxfun_per_parameter': 200,
               'xtol': 1e-30,
               'ftol': 1e-30,
               'full_output': True,
               'retall': False,
               'disp': False,
               'fmin_callback': None}
    
eis_func.run_fit(datafilepath, prmfilepath, **fit_options)
