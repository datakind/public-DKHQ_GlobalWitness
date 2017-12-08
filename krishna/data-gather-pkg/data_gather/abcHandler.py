#!/usr/bin/python3

"""
Handler Class


"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright '
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = ''
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = ''


import abc
from ValidationLogic import ValidationLogic

class abcHandler(metaclass=abc.ABCMeta):
    """
    Define an interface for handling requests.
    Implement the successor link.
    """

    def __init__(self, request, successor=None):
        self.request = request
        self.successor = ValidationLogic.isSuccessor(successor)

    def get_request(self):
        return self.request

    @abc.abstractmethod
    def handle(self):
        pass

if __name__ == "__main__":
    print("This is an abstract base class. No functionality available here.")
