'''Installer for storage package.'''

import setuptools


setuptools.setup(
    name='features',
    version='0.0.1',
    author='Daniel Duckworth',
    author_email='duckworthd@gmail.com',
    description='API for downloading features.',
    packages=['features'],
    include_package_data=True,
    install_requires=[
        'gevent',
        'earthengine-api',
        'GDAL',
        'geopandas',
        'numpy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
