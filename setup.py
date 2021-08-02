"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 1/6/21 2:37 pm
@desc  :
"""
from os import path as os_path
from setuptools import setup  # setuptools.setup encapsulates distutils.core.setup
# from distutils.core import setup
import re


target_pkg_name = 'fuzzytrees'
curr_dir = os_path.abspath(os_path.dirname(__file__))


def get_property(prop, pkg_name):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(pkg_name + '/__init__.py').read())
    return result.group(1)


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
    name='fdts',
    # Specify package information.
    packages=['fuzzytrees'],
    # package version.
    version=get_property(prop='__version__', pkg_name=target_pkg_name),
    license='MIT',
    # description of the package, shown on PyPI.
    description='An algorithm framework for developing algorithms based on fuzzy decision trees.',
    # long description of the package, from the README document.
    long_description=read_file('README.md'),
    keywords=['algorithm', 'fuzzy', 'fuzzy tree', 'decision tree', 'fuzzy algorithm'],
    # Specify the package document format as Markdown.
    long_description_content_type='text/markdown',
    author='Zhaoqing Liu, Anjin Liu, Jie Lu, Guangquan Zhang',
    author_email='Zhaoqing.Liu-1@student.uts.edu.au',
    url='https://github.com/ZhaoqingLiu/FuzzyTrees',
    download_url='https://github.com/ZhaoqingLiu/FuzzyTrees/archive/refs/tags/v_001.tar.gz',
    # python environment.
    python_requires='>=3.6.0',
    # Specify the dependencies that need to be installed.
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    classifiers=[
        # How mature is the project? Common values are
        #   2 - Pre-alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Release candidate (RC)
        #   6 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Who is project intended for?
        'Intended Audience :: Researchers, developers',
        'Topic :: Software Development :: Algorithm Framework',
        # License for this package.
        'License :: OSI Approved :: MIT License',
        # Natural language for the software package.
        'Natural Language :: English',
        # Python versions the package supports.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)


if __name__ == '__main__':
    print(get_property('__version__', target_pkg_name))
    print(read_file('README.md'))
    print(read_requirements('requirements.txt'))

