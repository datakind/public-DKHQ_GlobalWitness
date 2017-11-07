'''Installer for storage package.'''

import setuptools


setuptools.setup(
    name='storage',
    version='0.0.1',
    author='Daniel Duckworth',
    author_email='duckworthd@gmail.com',
    description='Library for storing image datasets.',
    packages=['storage'],
    include_package_data=True,
    install_requires=[
        'bcolz',
        'numpy',
        'pandas',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
