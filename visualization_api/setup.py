'''Installer for storage package.'''

import setuptools


setuptools.setup(
    name='visualization',
    version='0.0.1',
    author='Daniel Duckworth',
    author_email='duckworthd@gmail.com',
    description='Library for visualizing mining location predictions.',
    packages=['visualization'],
    include_package_data=True,
    install_requires=[
        'folium',
        'jupyter',
        'matplotlib',
        'numpy',
        'storage',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
