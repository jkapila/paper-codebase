from setuptools import setup
# from setuptools import find_packages
# pkgs = find_packages()

setup(
    name='relm',
    version='0.3',
    author="Jitin Kapila and Kumarjit Pathak",
    author_email="jitin.kapila@outlook.com",
    packages=['relm', ],
    license="GNU General Public License v3.0",
    description='Code for the concept RELM (Refer Paper: https://arxiv.org/pdf/1810.03198.pdf) ',
    url='http://github.com/jkapila/relm_code',
    zip_safe=False,
    requires=['numpy', 'pandas', 'cma'])
