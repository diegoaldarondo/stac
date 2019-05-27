"""Setup file for rat-stac."""
from distutils.core import setup

setup(
    name='rat-stac',
    version='0.1dev',
    packages=['rat-stac'],
    install_requires=['six >= 1.12.0',
                      'clize >= 4.0.3',
                      'pickle >= 4.0',
                      'absl-py >= 0.7.1',
                      'enum34',
                      'future',
                      'futures',
                      'glfw',
                      'lxml',
                      'numpy >= 1.16.13',
                      'pyopengl',
                      'pyparsing',
                      'h5py >= 2.9.0',
                      'scipy >= 1.2.1',
                      'yaml',
                      'opencv-python']
)
