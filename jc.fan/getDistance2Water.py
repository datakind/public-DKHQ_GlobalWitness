# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 16:48:47 2017
@author: junchuan fan

This module generates an 100*100 image patch for a mining site.

getDistance2WaterFeature(pcode,fp=water_fp)

"""

import json
import ee  # Earth Engine
import numpy as np
import pandas as pd
from pandas import DataFrame


import gdal
from geopandas import GeoDataFrame
from osgeo import ogr,osr



from joblib import Parallel,delayed
import multiprocessing as mp

import storage

#Initialize ee
ee.Initialize()

# path to water body shape file
water_fp="/home/fanj/MiningDetection/democratic_republic_of_the_congo_water/democratic_republic_of_the_congo_water.shp"


def getDistance2WaterFeature(pcode,fp=water_fp):
    '''
    input:
    pcode: the identifier for mining site
    
    output:
    an 100*100 image patch for this mining site. 
    Each pixel value represents the distance (meters) from this pixel to nearest water body
    
    If this image patch already exists, then load and return it;
    otherwise, create one image patch in the folder first, and return it. 
    
    '''
    
    #get the lat,lng from a site id
    ipis = ee.FeatureCollection('ft:1P1f-A2Sl44YJEqtD1FvA1z7QtDFsRut1QziMD-nV').getInfo()    
    lat=ipis['features']['pcode'==pcode]['geometry']['coordinates'][1]  
    lng=ipis['features']['pcode'==pcode]['geometry']['coordinates'][0]
    
    
    
    # Open an on-disk image dataset (may not exist yet).
    dataset = storage.DiskDataset("/tmp/dataset")

    # Add a new image to the dataset. Images are indexed by pcode and image
    # source (e.g. "distance2water"). Both are arbitrary strings.
   
    location_id = pcode
    source_id = "distance2water"
    
    metadata = {"bands": ["distance2water"]}
    if not dataset.has_image(location_id, source_id):
        image= calculateDistanceMatrix(lat,lng,fp)
        
        dataset.add_image(location_id, source_id, image, metadata)
        return image
    else:
        #print "image {}/{} already exists!".format(location_id, source_id)
        image = dataset.load_image(pcode, source_id)
        return image
        


def calculateDistanceMatrix(lat,lng,feature_fp):
    
    cpu_n=mp.cpu_count()
    
     

    result=Parallel(n_jobs=cpu_n)(delayed(Distance2NearestFeature)(x,y,feature_fp) for x,y in generateSiteMeshGrid(lat,lng))
    res_arr=np.asarray(result)
    image_matrix=res_arr.reshape((100,100))
    
    return image_matrix

        


def fromGeographic2ProjectCoor(lat,lng):
    
#Geographic coordinate system WGS84
    targetSRS = osr.SpatialReference()
    targetSRS.ImportFromEPSG(4326)
    
    #Projected coordinate system 
    targetPrjSRS = osr.SpatialReference()
    targetPrjSRS.ImportFromEPSG(4059)
    

    wktForm="Point ("+str(lng)+" "+str(lat)+")" 
    point=ogr.CreateGeometryFromWkt(wktForm,targetSRS)
    
    transform = osr.CoordinateTransformation(targetSRS,targetPrjSRS)
    point.Transform(transform)
    
    return point

def generateSiteMeshGrid(lat,lng):
    sitePnt=fromGeographic2ProjectCoor(lat,lng)
    
    x=sitePnt.GetPoint()[0]
    y=sitePnt.GetPoint()[1]
    x_min=x-1500
    x_max=x+1500
    y_min=y-1500
    y_max=y+1500
    xcoor_list=np.arange(x_min,x_max,30)
    ycoor_list=np.arange(y_min,y_max,30)
    
    #Mesh grid from left to right, bottom to top
    xx,yy=np.meshgrid(xcoor_list,ycoor_list)
    
    siteMesh=np.vstack([xx.ravel(),yy.ravel()])
    
    #return siteMesh
    
    # from top to bottom , the y coordinate decreases
    picDim=(100,100)
    picNum=100*100
    for i in range(picNum):
        yield (siteMesh[0][i],siteMesh[1][picNum-i-1])
        
        
        
        

def Distance2NearestFeature(x,y,feature_fp):
    '''
    input:
    (x,y): coordinates. 
    The underlying geographic reference system is EPSG 4059
    feature_fp: the path in which the shapefile of the feature is stored.     
        
    
    output: distance in meters
    '''

    
    #Geographic coordinate system WGS84
    targetSRS = osr.SpatialReference()
    targetSRS.ImportFromEPSG(4326)
    
    #Projected coordinate system 
    targetPrjSRS = osr.SpatialReference()
    targetPrjSRS.ImportFromEPSG(4059)
    

    wktForm="Point ("+str(x)+" "+str(y)+")" 
    point=ogr.CreateGeometryFromWkt(wktForm,targetPrjSRS)
    
      
    inDriver=ogr.GetDriverByName('ESRI Shapefile')
    feature_shp = inDriver.Open(feature_fp, 0)

    memDriver=ogr.GetDriverByName("MEMORY")
    memDataSource=memDriver.CreateDataSource('memData')

    #open the memory datasource with write access
    tmp=memDriver.Open('memData',1)


    #copy a layer to memory
    feature_mem=memDataSource.CopyLayer(feature_shp.GetLayer(),'f_name',['OVERWRITE=YES'])

    feature_layer=memDataSource.GetLayer('f_name')
    

    #Project both layer from geographic coordinate system to projected coordinate system
    #using EPSG:4059 (http://epsg.io/4059)
    transform = osr.CoordinateTransformation(targetSRS,targetPrjSRS)

    distList=[]
    for feature in feature_layer:
        geom=feature.GetGeometryRef()
        geom.Transform(transform)
        #name=feature.GetField('OBJECTID')
       # print(name,point.Distance(geom))
        distList.append(point.Distance(geom))
        
       # print(geom.Centroid().ExportToWkt(),spatialRef)
    
   # print(sorted(distList))
    
    return min(distList)