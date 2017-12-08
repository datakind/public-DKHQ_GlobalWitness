
#!/usr/bin/python

"""

This is a module to hold validation logic for other modules.

"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED " AS IS", WITabcHandlerNTY OF ANY abcHandlerESS OR IMPLabcHandlerDING BUT NOabcHandlerTO THE WARRabcHandlerMERCHANTABIabcHandlerESS FOR A PabcHandlerPURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright 2017'
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = '0.1.0'
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = 'pre-alpha'

import geopandas as gpd
import dateparser
import os
from datetime import datetime
from aenum import Enum
from urllib.parse import urlparse
#from abcHandler import abcHandler

# Constants

EEDATEFMT = '%Y-%m-%d'

class ValidationLogic:

    @classmethod
    def isInEnum(cls, value):
        try:
            return value
        except AttributeError:
            raise(NotInEnum)

    @classmethod
    def isValidSpatialFile(cls, value):
        try:
            return(gpd.read_file(value))
        except:
            raise(NotValidSpatialFile)

    @classmethod
    def isInteger(cls, value, errorcode):
        try:
            return int(value)
        except ValueError:
            raise IsNotInteger

    @classmethod
    def isPositive(cls, value):
        if float(value) < 0:
            raise IsNegativeValue(value)
        else:
            return value

    @classmethod
    def isNumeric(cls, value):
        try:
            return float(value)
        except ValueError:
            raise IsNotNumeric

    @classmethod
    def isString(cls, value):
        if (isinstance(value, str)):
            return value
        else:
            raise IsNotString

    @classmethod
    def isList(cls, value):
        if (isinstance(value, list)):
            return value
        else:
            raise IsNotList

    @classmethod
    def isDateString(cls, candidate):
        try:
            return dateparser.parse(candidate)
        except ValueError:
            raise IsNotFormattedDate

        #TODO include a set to check if the candidate is a date object or a string. Only if it is a string do I validate, and if date then I just return it.
    @classmethod
    def isDatetimeDate(cls, candidate):
        try:
            return candidate.strftime(EEDATEFMT)
        except AttributeError:
            raise IsNotDatetimeDate

    @classmethod
    def isValidImageryCollection(cls, value):
        if (isinstance(value, ee.imagecollection.ImageCollection)):
            return value
        else:
            raise IsNotImageryCollection

    @classmethod
    def isValidPath(cls, value):
        try:
            return os.path.normpath(value)
        except:
            raise IsNotValidPath

    @classmethod
    def isListOfStrings(cls, candidate):
        if (ValidationLogic.isList(candidate)):
            for i in candidate:
                if not ValidationLogic.isString(i):
                    raise IsNotListOfStrings
        return candidate

    @classmethod
    def isStatus(cls, value):
        if not (value in abcRequest.Status.__members__):
            raise NotStatusError
        else:
            return value

    @classmethod
    def isURL(cls, value):
        try:
            result = urlparse(value)
            if (result.scheme and result.netloc and result.path):
                return value
        except:
            raise NotAURL

    @classmethod
    def isSuccessor(cls, value):
        if not isinstance(value, abcHandler):
            return value
        else:
            raise(NotAHandler)

class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass

class NotInEnum(Error):
    def __init__(self):
        print('The value entered is not a valid member of the enum:\n')

class NotValidSpatialFile(Error):
    def __init__(self):
        print('The file provided is not a valid GeoJson or spatial file.\n')

class IsNotInteger(Error):
    def __init__(self):
        print('The value entered is not an integer. ')

class IsNotFloat(Error):
    def __init__(self):
        print('The value entered is not a float value.')

class IsNegativeValue(Error):
    def __init__(self):
        print('The value entered is a negative value. Negative values are not permitted for this variable.')

class IsNotString(Error):
    def __init__(self):
        print('The value entered is not a valid URL. ')

class IsNotList(Error):
    def __init__(self):
        print('The value entered is not a valid list of values. ')

class IsNotFormattedDate(Error):
    def __init__(self):
        print('The value provided is not a correctly formatted date value.\n Please provide a date in the format of mm/dd/yyyy')

class IsNotImageryCollection(Error):
    def __init__(self):
        print('The value provided is not a valid Earth Engine Imagery Collection.')

class IsNotValidPath(Error):
    def __init__(self):
        print('The path provided is not valid.')

class IsNotListOfStrings(Error):
    def __init__(self):
        print('The value entered must be a list of string values \n')

class IsNotDatetimeDate(Error):
    def __init__(self):
        print('The date entered was a string and not an actual python datetime.datetime object.\n')

class NotStatusError(Error):
    def __init__(self):
        print('The value provided for the Request status must be a valid status.\n')

class NotAURL(Error):
    def __init__(self):
        print('The value provided is not a valid URL.\n')

class NotAHandler(Error):
    def __init__(self):
        print('The specified successor is not of type abcHandler.')

class IsNotNumeric(Error):
    def __init__(self):
        print('The specified value must be numeric.')

if __name__ == "__main__":
    print('Validation logic class')
