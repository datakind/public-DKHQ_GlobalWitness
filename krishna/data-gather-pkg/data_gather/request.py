
#!/usr/bin/python

"""
Basic request class


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


from abcHandler import Handler
import uuid
from enum import Enum

class Request():

    class State(Enum):
        open = 'open'
        close = 'closed'
        rejected = 'rejected'

    class Composites(Enum):
        none = 'none'
        monthly = 'monthly'
        quarterly = 'quarterly'
        yearly = 'yearly'

    def __init__(reqCollection, bands, startdate, enddate, compositeFlag):
        self.id = uuid.uuid5()  # unique id for request
        self.status = Request.State.open # request status
        self.requestCollection = reqCollection # satellite imagery collection
        self.requestBands = [] # bands from imagery collection
        self.startdate = startdate # state date for imagery request
        self.enddate = enddate # end date for imagery request
        self.compositeFlag = compositeFlag # composite images or raw daily images
        self.status = Request.State.open #
        self.urllist = []


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


class Tests():
    def test_t1(self):
        pass


def main()


if __name__ == "__main__":
    main()
