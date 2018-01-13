#!/usr/bin/python

"""

Point Imagery Request Builder

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


from abcRequestBuilder import abcRequestBuilder
from pointImageryRequest import PointImageryRequest
from pointImageryRequest import PointImageryRequestStatusCodes
from HandlerAssignEEEngineToRequest import HandlerAssignEEEngineToRequest
from HandlerLoadPointData import HandlerLoadPointData
from HandlerSetRequestStatus import HandlerSetRequestStatus
from HandlerSetRequestDates import HandlerSetRequestDates
from HandlerSetRadius import HandlerSetRadius


class BuilderPointImageryRequest(abcRequestBuilder):


    def __init__(self, settings=None):
        super().__init__()
        self.request = PointImageryRequest()
        self.request.settings = settings

    def originate_request(self):
        HandlerAssignStatusEnumToRequest(self.request).handle()
        HandlerSetRequestDates(self.request).handle()
        HandlerSetRadius(self.request).handle()
        HandlerSetImageCollection(self.request).handle()

    def assign_data(self):

        HandlerLoadPointData(self.request).handle()

    def validate_request(self):
    
        #TODO create handler to confirm epsg 4236 or to convert to epsg 4236
        HandlerSetRequestStatus(self.request).handle(PointImageryRequestStatusCodes.READYTOPROCESS)
      



class Tests:

    def test_1(self):
        assert 1 == 1




if __name__ == "__main__":
    print("This is the builder class for the point imagery request")
