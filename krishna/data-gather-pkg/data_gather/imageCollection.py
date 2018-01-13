
#!/usr/bin/python

"""
Imagery Collection Data Classes

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

from datetime import datetime
from ValidationLogic import ValidationLogic



class ImageCollection():

    def __init__(self, id, bands, startdate, enddate, resolution):
        self.id = ValidationLogic.isString(id)
        self.bands = ValidationLogic.isListOfStrings(bands)
        self.startdate = ValidationLogic.isDateString(startdate)
        self.enddate = ValidationLogic.isDateString(enddate)
        self.resolution = ValidationLogic.isPositive(resolution)


    def get_id(self):
        return self.id

    def get_bands(self):
        return self.bands

    def get_startdate(self):
        return self.startdate

    def get_enddate(self):
        return self.enddate

    def get_resolution(self):
        return self.resolution

    def set_bands(self, candidate):
        self.bands = ValidationLogic.validate_list_of_string(candidate)

    def set_startdate(self, candidate):
        self.startdate = ValidationLogic.is_date_string(candidate)

    def set_enddate(self, candidate):
        self.enddate = ValidationLogic.is_date_string(candidate)

    def set_resolution(self, candidate):
        self.resolution = ValidationLogic.isPositive(candidate)

    def is_band_in_bands(self,candidate):
        if (ValidationLogic.isstring(candidate) in self.bands):
            return True
        else:
            return False

    def is_date_in_range(self, candidate):
        candidate = ValidationLogic.is_date_string(candidate)

        if (self.startdate <= candidate <= self.enddate):
            return True
        else:
            return False


class Tests():

    test = ImageCollection('LANDSAT/LC08/C01/T1', ['B1','B2','B3','QA'], '05/01/1998', '12/31/2017', 30)

    def test_create_imagecollection_class(self):
        test = ImageCollection('LANDSAT/LC08/C01/T1', ['B1','B2','B3','QA'], '05/01/1998', '12/31/2017')
        assert isinstance(test, ImageCollection)

    def test_get_starttdate(self):
        assert self.test.get_startdate() == datetime(1998, 5, 1).date()

    def test_get_enddate(self):
        assert self.test.get_enddate() == datetime(2017, 12, 31).date()

    def test_get_bands(self):
        assert self.test.bands == ['B1','B2','B3','QA']

    def test_get_id(self):
        assert self.test.id == 'LANDSAT/LC08/C01/T1'

    def test_band_in_bands(self):
        assert self.test.is_band_in_bands('B3') == True

    def test_date_in_range(self):
        assert self.test.is_date_in_range('11/12/2016')


if __name__ == "__main__":
    main()
