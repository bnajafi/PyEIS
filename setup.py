try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import pyeis

setup(
    name='PyEIS',
    version=pyeis.__version__,
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/MilanSkocic/PyEIS.git',
    download_url='https://github.com/MilanSkocic/PyEIS.git',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Milan Skocic',
    author_email='milan.skocic@gmail.com',
    maintainer='Milan Skocic',
    maintainer_email='milan.skocic@gmail.com',
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    description='Contains core functions for computing and fitting experimental data '
                'from Electrochemical Impedance Spectroscopy.',
    long_description=pyeis.__doc__,
    install_requires=['numpy>=1.8', 'scipy>=0.14', 'sympy>=0.7.4', 'matplotlib>=1.3.0','docutils>=0.3',
                      'prettytable>=0.7.2'],
    classifiers="""Development Status :: 1 - Planning
    Intended Audience :: Science/Research
    Intended Audience :: Electrochemistry Engineers
    License :: GNU GENERAL PUBLIC LICENSE
    Programming Language :: Python
    Programming Language :: Python :: 3
    Topic :: Scientific/Engineering
    Operating System :: Microsoft :: Windows
    Operating System :: POSIX
    Operating System :: Unix
    Operating System :: MacOS"""
)
