

import ee
ee.Initialize()

coords = [0.32153, 28.69916]
newFil = ee.Geometry.Point(coords).buffer(300)
data = ee.ImageCollection('LANDSAT/LC8_L1T_32DAY_TOA').select('B8').filterDate('2016-1-1', '2016-5-30')
data.filterBounds(newFil)

def clipper(image):
  return image.clip(newFil.bounds())

data.map(clipper)
req = data.getInfo()
boundary = ee.Geometry(newFil.bounds().getInfo()).toGeoJSONString()

def downloader(image):
    url = ee.data.makeDownloadUrl(
        ee.data.getDownloadId({
            'image': image.serialize(),
            'scale': 30,
            'filePerBand': 'false',
            'name': 'test',
            'region': boundary,
        }))
    print(url)



downloader(ee.Image(data.getInfo()['features'][1]['id']))
