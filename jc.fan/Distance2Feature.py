# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 16:48:47 2017

@author: junchuan fan
"""

try:
  from osgeo import ogr,osr
  print('Import osgeo worked.  Hurray!\n')
except:
  print('Import of osgeo failed\n\n')



def Distance2NearestFeature(lat,lng,feature_fp):
    '''
    input: (lat,lng) coordinates. 
    The underlying geographic reference system is EPSG 4326, a.k.a, WGS84 .
    feature_fp: the path in which the shapefile of the feature is stored.     
        
    
    output: distance in meters
    '''

    
    #Geographic coordinate system WGS84
    targetSRS = osr.SpatialReference()
    targetSRS.ImportFromEPSG(4326)
    
    #Projected coordinate system 
    targetPrjSRS = osr.SpatialReference()
    targetPrjSRS.ImportFromEPSG(4059)
    

    wktForm="Point ("+str(lng)+" "+str(lat)+")" 
    point=ogr.CreateGeometryFromWkt(wktForm,targetSRS)
    
      
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
    point.Transform(transform)
    
    

 
    
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