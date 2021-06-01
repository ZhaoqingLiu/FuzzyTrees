"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 1/6/21 2:37 pm
@desc  :
"""
from os import path as os_path
from setuptools import setup

import fdt

curr_dir = os_path.abspath(os_path.dirname(__file__))


# Read a file.
def read_file(filename):
    with open(os_path.join(curr_dir, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


# Find all packages.
def find_packages():
    pass


# Get all dependencies.
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(
    # package name.
    name='fuzzy_trees',
    # python environment.
    python_requires='>=3.6.0',
    # package version.
    version=fdt.__version__,
    # description of the package, shown on PyPI.
    description='An algorithm framework integrating fuzzy decision trees and fuzzy ensemble trees.',
    # long description of the package, from the README document.
    long_description=read_file('README.md'),
    # Specify the package document format as Markdown.
    long_description_content_type='text/markdown',
    author='Zhaoqing.Liu',
    author_email='Zhaoqing.Liu-1@student.uts.edu.au',
    url='https://github.com/ZhaoqingLiu/FuzzyDecisionTrees',
    # Specify package information.
    packages=[
        'fuzzy_decision_tree_proxy',
        '',
        ''
    ],
    # Specify the dependencies that need to be installed.
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    license='MIT',
    keywords=['fuzzy', 'decision tree', 'gradient boosting'],
    classifiers=[
        'Intended Audience :: Researchers, developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
