#!/usr/bin/python

"""

Concrete points request class

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

import abc
from aenum import Enum, extend_enum
import ee
from .AdapterSpecifyImageryCollection import AdapterSpecifyImageryCollection
from .AdapterDateFilter import AdapterDateFilter
from .AdapterPointBoundingBox import AdapterPointBoundingBox

ee.Initialize()

class abcEarthEngineProcessor(metaclass=abc.ABCMeta):

    def __init__(self, request):
        self.request = request
        self.imageryCollection = None
        self.featureCollection = None


    def set_imageryCollection(self):
        self.imageryCollection = AdapterSpecifyImageryCollection.request(self.get_request().get_imageryCollection())


    def set_featureCollection(self):
        pass

    def set_dateFilterToRequestDates(self):

        self.imageryCollection = AdapterDateFilter.request(self.get_imageryCollection(),
                                                           self.get_request().get_string_startdate,
                                                           self.get_string_enddate)

    def set_dateFilterToCustomDates(self, strStartDate, strEndDate):

        self.imageryCollection = AdapterDateFilter.request(self.get_imageryCollection(), strStartDate, strEndDate)

    def set_boundaryFilter(self, coords):

        self.imageryCollection = AdapterPointBoundingBox.request(self.get_imageryCollection(),
                                                                 coords,
                                                                 self.get_request().get_radius())




    def get_imageryCollection(self):
        return self.imageryCollection

    def get_featureCollection(self):
        return self.featureCollection

    def get_request(self):
        return self.request


    @abc.abstractmethod
    def process(self):
        pass





class ValidationLogic:

    @classmethod
    def isnotinteger(cls, value):
        try:
            return int(value)
        except ValueError as e:
            raise IsNotInteger(e)

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class Error1(Error):
    def __init__(self, evalue):
        print('The value entered is invalid: ' + str(evalue))

def main():
    pass

if __name__ == "__main__":
    main()
