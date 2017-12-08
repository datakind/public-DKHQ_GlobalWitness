
#!/usr/bin/python

"""

Point imagery request class

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

from abcRequest import abcRequest

class RequestEEPointImagery(abcRequest):

    def __init__(self, settings):
        super().__init__(settings)
        self.latitude = self.settings['latitude']
        self.longitude = self.settings['longitude']
        self.epsg = self.settings['epsg']
        self.radius = self.settings['radius']
        self.startdate = self.settings['startdate']
        self.enddate = self.settings['enddate']
        self.bands = self.settings['bands']
        self.ee = None

    def get_latitude(self):
        return self.latitude

    def get_longitude(self):
        return self.longitude

    def get_epsg(self):
        return self.epsg

    def get_radius(self):
        return self.radius

    def get_startdate(self):
        return self.startdate

    def get_enddate(self):
        return self.enddate

    def get_bands(self):
        return self.bands








if __name__ == "__main__":
    main()
