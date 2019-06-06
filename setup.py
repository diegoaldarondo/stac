"""Setup file for stac."""
from setuptools import setup

setup(
    name='stac',
    version='0.0.1',
    install_requires=['six >= 1.12.0',
                      'clize >= 4.0.3',
                      'absl-py >= 0.7.1',
                      'enum34',
                      'future',
                      'futures',
                      'glfw',
                      'lxml',
                      'numpy',
                      'pyopengl',
                      'pyparsing',
                      'h5py >= 2.9.0',
                      'scipy >= 1.2.1',
                      'pyyaml',
                      'opencv-python']
)
