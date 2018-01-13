
"""
Created on Sat Nov 25 16:18:47 2017
@author: junchuan fan

This module generates an image patch for a mining site.
The dimension of the image patch is determined by the metadata returned by ee.utils API. 


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

import features
import storage

#Initialize ee
ee.Initialize()

# path to water body shape file
water_fp="/home/fanj/MiningDetection/ON-MiningDetection/jc.fan/democratic_republic_of_the_congo_water/democratic_republic_of_the_congo_water.shp"


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
    source_id = "distance_to_ground_feature"
    
    metadata = {"bands": "water"}
    if not dataset.has_image(location_id, source_id):
        image= calculateDistanceMatrix(lat,lng,fp)
        
        dataset.add_image(location_id, source_id, image, metadata)
        return image
    else:
        #print "image {}/{} already exists!".format(location_id, source_id)
        image = dataset.load_image(pcode, source_id)
        return image
        


def calculateDistanceMatrix(lat,lng,feature_fp,patch_size=100,meters_per_pixel=30):
    
    cpu_n=mp.cpu_count()
    
     
    image_patch=features.feature_handlers.ee_utils.download_pixel_centers((lat,lng),patch_size,meters_per_pixel)[0]
    
    image_patch_shape=image_patch.shape
    
    result=Parallel(n_jobs=cpu_n)(delayed(Distance2NearestFeature)(x,y,feature_fp) for x,y 
                                  in generateImagePatchCoors(lat,lng,image_patch))
    res_arr=np.asarray(result)
    
    image_matrix=res_arr.reshape((image_patch_shape[0],image_patch_shape[1]))
    
    return image_matrix

        

def project_geographic_coordinate(lat,lng):
    '''Take lat,lng pair, and project them into x,y in EPSG 4059 coordinate system'''
    prjCoor=fromGeographic2ProjectCoor(lat,lng)
    return prjCoor.GetPoint()[0],prjCoor.GetPoint()[1]
    
def generateImagePatchCoors(lat,lng,image_patch):
    '''
    yield x,y coordinate pairs for each pixel of this mining site's image patch
    '''

    image_patch_shape=image_patch.shape
    for i in range(image_patch_shape[0]):
        for j in range(image_patch_shape[1]):
            x,y=project_geographic_coordinate(image_patch[i,j,0],image_patch[i,j,1])
            yield x,y
    #      print i
    
    

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

#if __name__=='__main__':
#    ipis = ee.FeatureCollection('ft:1P1f-A2Sl44YJEqtD1FvA1z7QtDFsRut1QziMD-nV').getInfo()  

#    for feature in ipis['features']:
#        print feature['properties']['pcode']
#        pcode=feature['properties']['pcode']
#        getDistance2WaterFeature(pcode)
