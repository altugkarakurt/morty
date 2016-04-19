#!/usr/bin/env python

from setuptools import setup

setup(name='morty',
version='1.0',
    author='Altug Karakurt',
    author_email='altugkarakurt AT gmail DOT com',
    license='agpl 3.0',
    description='Python library for mode and tonic estimation.',
    url='https://github.com/altugkarakurt/morty',
    packages=['morty'],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
)
