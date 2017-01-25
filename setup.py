#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup


setup(name='morty',
      version='v1.3.0',
      author='Altug Karakurt',
      author_email='altugkarakurt AT gmail DOT com',
      license='agpl 3.0',
      description='Python library for mode and tonic estimation.',
      url='https://github.com/altugkarakurt/morty',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "numpy>=1.9.0",
          "scipy>=0.17.0",
          "matplotlib>=1.5.1"
      ],
      )
