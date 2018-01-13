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
from HandlerLoadPointData import HandlerLoadPointData
from HandlerDateFilter import HandlerDateFilter
from HandlerPointBoundingBox import HandlerPointBoundingBox
from HandlerPointClip import HandlerPointClip
from HandlerPointDownloadURL import HandlerPointDownloadURL
from HandlerURLDownloader import HandlerURLDownloader
from BuilderPointImageryRequest import BuilderPointImageryRequest
from ValidationLogic import ValidationLogic
from HandlerEESimplePointImageryProcessor import HandlerEESimplePointImageryPointProcessor
from PackageConstants import imgCollections


click.group()
@click.option('--startdate', default = None, type=str)
@click.option('--enddate', default = None, type=str)
@click.option('--directory', default = None, type=click.Path())
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
    ctx.obj['startdate'] = ValidationLogic.isDateString(startdate)
    ctx.obj['enddate'] = ValidationLogic.isDateString(enddate)


@click.command()
@click.argument('collection', type=click.Choice(['Landsat8', 'Landsat7', 'Landsat5']))
@click.argument('filename', type=click.Path(exists=True))
@click.argument('radius', type=int)
@click.pass_context
def SimplePointImageryRequest(ctx,
                              collection,
                              filename,
                              radius):

    """Download raw point imagery patches from EE collection."""


    ctx.obj['radius'] = ValidationLogic.isPositive(radius)
    ctx.obj['collection'] = imgCollections.get(collection)
    settings = ctx.obj

    request_queue = build_queue(filename, settings)
    process_queue(request_queue)
    download_queue = download_queue(process_queue)

def build_request(builder, dictSettings):

    # TODO this might not work on the builder() since it is a variable. Fix later.
    tempRequest = builder(dictSettings)
    director = DirectorRequestBuilder()
    director.construct(tempRequest, dictSettings)
    newRequest = tempRequest.request
    return newRequest

def InvokerSimplePointImageryRequest(request):

    pass
    PointProcessor(request).handle()
    Download(request).handle()

def InvokerImageryDownloader(request):
    pass


class RequestTypes(Enum):
    SIMPLEPOINTIMAGERY = 1
    DIVAGIS = 2
    COMPOSITEDPOINTIMAGERY = 3



cli.add_command(SimplePointImageryRequest)

if __name__ == "__main__":
    cli(obj={})
