#!/usr/bin/python

"""

Simple point imagery request for Earth Engine

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

from aenum import Enum, extend_enum
import ee
from .AdapterSpecifyImageryCollection import AdapterSpecifyImageryCollection
from .AdapterDateFilter import AdapterDateFilter
from .AdapterPointBoundingBox import AdapterPointBoundingBox
from .abcEarthEngineProcessor import abcEarthEngineProcessor


ee.Initialize()

class EarthEngineSimplePointImageryProcessor(abcEarthEngineProcessor):

    def process(self):

        self.add_metadata_columns_to_request_data()

        for index, row in self.get_request().get_data_iterator():
            self.set_imageryCollection()
            self.set_dateFilterToRequestDates()
            # TODO check the order of y and x for proper filter.
            coords = [row.Geometry.y, row.Geometry.x]
            self.set_boundaryFilter(coords)

            p = ee.Geometry.Point(coords).buffer(self.request.get_radius())

            def clipper(image):
                return image.clip(p.bounds())

            boundary = ee.Geometry.Polygon(p.bounds().getInfo().coordinates[0]).toGeoJSONString();

            self.get_imageryCollection().map(clipper)


            url = ee.data.makeDownloadUrl(
                    ee.data.getDownloadId({
                        'image': image.serialize(),
                        'scale': '%d' % self.get_request().get_resolution(),
                        'crs': 'EPSG:4326',
                        'filePerBand': 'false',
                        'name': filename,
                        'region': boundary,
                    }))


    def add_metadata_columns_to_request_data(self):
        self.get_request().add_column_to_data('source_id')
        self.get_request().add_column_to_data('bands')
        self.get_request().add_column_to_data('imagery_startdate')
        self.get_request().add_column_to_data('imagery_enddate')
        self.get_request().add_column_to_data('download_date')

