from distutils.core import setup

setup(
    name='PyEIS',
    version='0.2',
    py_modules=['eis_functions', 'circuit_decomposition', 'errors'],
    url='https://github.com/MilanSkocic/PyEIS.git',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Milan',
    author_email='milan.skocic@gmail.com',
    description='Contains core functions for computing and fitting experimental data '
                'from Electrochemical Impedance Spectroscopy.',
)
