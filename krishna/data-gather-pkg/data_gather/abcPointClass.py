
#!/usr/bin/python

"""

Abc for Point class

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
from ValidationLogic import ValidationLogic


class abcPoint(metaclass=abc.ABCMeta):

    def __init__(self):
        self.id = None
        self.latitude = None
        self.longitude = None
        self.epsg = None
        self.download_url = None
        self.downloadSettings = {}
        self.startdate = None
        self.enddate = None
        self.metadata = {}

    def get_latitude(self):
        return self.latitude

    def get_longitude(self):
        return self.longitude

    def get_epsg(self):
        return self.epsg

    def get_id(self):
        return self.id

    def get_startdate(self):
        return self.startdate

    def get_enddate(self):
        return self.enddate

    def get_download_url(self):
        return self.download_url

    def get_downloadSettings(self):
        return self.downloadSettings

    def set_latitude(self, candidate):
        self.latitude = ValidationLogic.isNumeric(candidate)

    def set_longitude(self, candidate):
        self.longitude = ValidationLogic.isNumeric(candidate)

    def set_epsg(self, candidate):
        self.epsg = ValidationLogic.isPositive(candidate)

    def set_startdate(self, candidate):
        self.startdate = ValidationLogic.isDatetimeDate(candidate)

    def set_enddate(self, candidate):
        self.enddate = ValidationLogic.isDatetimeDate(candidate)

    def set_download_url(self, candidate):
        self.download_url = ValidationLogic.isURL(candidate)

    def add_download_setting(self, key, candidate):
        self.downloadSettings[key] = candidate

    @abc.abstractmethod
    def set_point_id(self):
        pass

    @abc.abstractmethod
    def set_point_metadata(self):
        pass

if __name__ == "__main__":
    print('Abstract base class for points')
