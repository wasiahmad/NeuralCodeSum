#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='c2nl',
    version='0.1.0',
    description='Code to Natural Language Generation',
    long_description=readme,
    license=license,
    python_requires='>=3.6',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)
