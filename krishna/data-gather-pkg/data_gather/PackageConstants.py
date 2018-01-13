
#!/usr/bin/python

"""

Package constants

"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright 2017'
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = '0.1.0'
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = 'pre-alpha'

from aenum import Enum


class RequestTypes(Enum):
    SIMPLEPOINTIMAGERY = 1
    DIVAGIS = 2
    COMPOSITEDPOINTIMAGERY = 3



class RequestStatusCodes(Enum):
    CLOSED = 0
    CREATED = 1
    QUEUED = 2
    PROCESSING = 3
    COMPLETED = 4
    REJECTED = 5
    ERROR = 6


imgCollections = {'Landsat8' : ImageCollection('LANDSAT/LC08/C01/T1',
                                                      ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA'],
                                                      '04/13/2011',
                                                      '10/07/2017',
                                                       30),
                        'Landsat7' : ImageCollection('LANDSAT/LE07/C01/T1',
                                                       ['B1','B2','B3','B4','B5','B6','B7'],
                                                      '01/01/1999',
                                                      '09/17/2017',
                                                       30),
                        'Landsat5' : ImageCollection('LANDSAT/LT05/C01/T1',
                                                      ['B1','B2','B3','B4','B5','B6','B7'],
                                                      '01/01/1984',
                                                      '05/05/2012',
                                                       30),
                        'Sentinel2msi' : ImageCollection('COPERNICUS/S2',
                                                          ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','QA10','QA20','QA60'],
                                                          '01/23/2015',
                                                          '10/20/2017',
                                                          30),
                        'Sentinel2sar' : ImageCollection('COPERNICUS/S1_GRD',
                                                         ['VV', 'HH',['VV', 'VH'], ['HH','HV']],
                                                         '10/03/2014',
                                                         '10/20/2017',
                                                         30),
                        'ModisThermalAnomalies' : ImageCollection('MODIS/006/MOD14A1',
                                                                  ['FireMask', 'MaxFRP','sample', 'QA'],
                                                                  '02/18/2000',
                                                                  '10/23/2017',
                                                                  30)
    }

if __name__ == "__main__":
    print('set of package constants.')
