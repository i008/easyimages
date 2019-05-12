#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip < 10
    from pip.req import parse_requirements
import os

__version__ = '1.01'

with open('README.MD') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

dir_path = os.path.dirname(os.path.realpath(__file__))
req_path = os.path.join(dir_path, 'requirements.txt')
install_reqs = parse_requirements(req_path, session='hack')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    author="Jakub Cieslik",
    author_email='kubacieslik@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Images made easy",
    install_requires=reqs,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='easyimages',
    name='easyimages',
    packages=find_packages(include=['easyimages']),
    setup_requires=reqs,
    test_suite='tests',
    tests_require=reqs,
    url='https://github.com/i008/easyimages',
    version=__version__,
    zip_safe=False,
)
