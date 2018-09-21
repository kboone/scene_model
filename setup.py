#!/usr/bin/env python

from setuptools import setup

setup(
    name='scene_model',
    version='0.1',
    description='Scene Modeling Package',
    author='Kyle Boone',
    author_email='kboone@berkeley.edu',
    packages=['scene_model'],
    scripts=['scripts/extract_star2.py', 'scripts/subtract_psf2.py'],
)
