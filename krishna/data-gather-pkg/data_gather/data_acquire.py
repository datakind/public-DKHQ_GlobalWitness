# -*- coding: utf-8 -*-

"""Console script for dk_earth_engine_downloader."""

import click
import dateparser
import os
from enum import Enum
from imageCollection import ImageCollection
from DirectorRequestBuilder import DirectorRequestBuilder
from Invoker import Invoker
from HandlerSpecifyImageryCollection import HandlerSpecifyImageryCollection
from CommandSimplePointImageryRequest import CommandSimplePointImageryRequest
from HandlerSetRequestDatesFullSatelliteDateRange import HandlerSetRequestDatesFullSatelliteDateRange
from HandlerLoadPointData import HandlerLoadPointData
from HandlerDateFilter import HandlerDateFilter
from HandlerPointBoundingBox import HandlerPointBoundingBox
from HandlerPointClip import HandlerPointClip
from HandlerPointDownloadURL import HandlerPointDownloadURL
from HandlerURLDownloader import HandlerURLDownloader
from BuilderPointImageryRequest import BuilderPointImageryRequest
from ValidationLogic import ValidationLogic
from HandlerEESimplePointImageryProcessor import HandlerEESimplePointImageryPointProcessor


@click.group()
@click.option('--startdate', type=str)
@click.option('--enddate', type=str)
@click.option('--directory', type=click.Path())
@click.pass_context
def cli(ctx,
        startdate,
        enddate,
        directory):

    """
    This script will download satellite imagery from Google Earth Engine.
    The user must specify a spatial data file containing points, and
    and imagery collection from the list of collections below. The
    application will then connect to earth engine and download imagery patches
    that match the point coordinates and request specifications.

    List of Imagery Collections:\n
    Landsat8:      Landsat 8 imagery at 30m resolution\n
    Landsat7:      Landsat 7 imagery at 30m resolution\n


    Request types include:\n
    SimplePointRequest:      Download raw image patches from specified\n
                             collection.\n

    CompositedPointRequest:  Download image composites from specified\n
                             collection.\n

    """

    ctx.obj['directory'] = directory
    ctx.obj['startdate'] = startdate
    ctx.obj['enddate'] = enddate


@click.command()
# TODO get the choice to work on the collection argument.
# TODO fix the issue with `Missing Argument directory` not showing up. 
@click.argument('collection', type=click.Choice(['Landsat8', 'Landsat7', 'Landsat5']))
@click.argument('filename', type=click.Path(exists=True))
@click.argument('radius', type=int)
@click.pass_context
def SimplePointImageryRequest(ctx,
                              collection,
                              filename,
                              radius):

    """Download raw point imagery patches from EE collection."""

    ctx.obj['filename'] = filename
    ctx.obj['radius'] = ValidationLogic.isPositive(radius)
    ctx.obj['collection'] = collection
    ctx.obj['statusList'] = PointImageryRequestStatusCodes
    settings = ctx.obj

    request = build_request(BuilderPointImageryRequest, settings)
    InvokerPointProcessorSimplePointImageryRequest(request)
    InvokerImageryDownloader(request)


def build_request(builder, argdict):

    # TODO this might not work on the builder() since it is a variable. Fix later.
    tempRequest = builder(argdict)
    director = DirectorRequestBuilder()
    director.construct(tempRequest)
    newRequest = tempRequest.request
    return newRequest

def registerSatelliteImageryCollections():

    imagecollections = {'Landsat8' : ImageCollection('LANDSAT/LC08/C01/T1',
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

    return imagecollections

def InvokerSimplePointImageryRequest(request):

    handlers = [HandlerEESimplePointImageryPointProcessor
                InvokerImageryDownloader
                ]

    invoker = Invoker()

    for c in handlers:
        invoker.store_command(c(request).handle())


    invoker.execute_commands()


def InvokerImageryDownloader(request):
    pass


class RequestTypes(Enum):
    SIMPLEPOINTIMAGERY = 1
    DIVAGIS = 2
    COMPOSITEDPOINTIMAGERY = 3

class PointImageryRequestStatusCodes(Enum):
    CLOSED = 0
    CREATED = 1
    READYTOPROCESS = 2
    PROCESSING = 3
    READYTODOWNLOAD = 4
    COMPLETED = 5


cli.add_command(SimplePointImageryRequest)

if __name__ == "__main__":
    cli(obj={})
