#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='7l-common',
    url='https://7learnings.com',
    description='Shared 7Learnings code and models',
    install_requires=[],
    extras_require={'tests': ['pytest', 'black', 'flake8']},
    packages=find_packages(exclude=["tests"]),
)
