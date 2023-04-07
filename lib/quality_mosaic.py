import ee

K = 1000
OFFSET = -0.5
MULT = 100
CLOUD_FILTER = 10
CLD_PRB_THRESH = 20


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability').rename('prob')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def addDate(image):
    #Subtract to get difference, take absolute, then negate (because qualityMosaic only does ascending), then convert to days
    deltaDays = DOI.millis().subtract(image.date().millis()).abs().divide(ee.Number(8.64e+7)).toInt()
    return image.addBands(ee.Image(deltaDays).rename('date').toInt())

def addFScore(image, DOI):
    """
    image: ee.Image type
    DOI: ee.Date type, e.g. ee.Date('2019-01-01') 
    """
    # img_cloud = add_cloud_bands(image) #now it had a band called 'prob'
    image = add_cloud_bands(image)
    
    #Subtract to get difference, take absolute, then negate (because qualityMosaic only does ascending), then convert to days
    deltaDays = DOI.millis().subtract(image.date().millis()).abs().divide(ee.Number(8.64e+7)).toInt()
    image = image.addBands(ee.Image(deltaDays).rename('delta').toInt())
    
    # image.addBands(ee.Image(deltaDays)).rename('deltaDays')
    # image = image.addBands(ee.Image(deltaDays).rename('delta'))
    prob = image.select('prob').divide(100)
    # image.addBands(prob).rename('prob')
    
    x = ee.Image(deltaDays.add(1).log10()).add(1).multiply(ee.Image(K).pow(prob.add(ee.Number(OFFSET)))).multiply(ee.Image(MULT))
    # x = ee.Image(deltaDays.log10()).toInt64().multiply(-1)
    return image.addBands(x.multiply(ee.Number(-1)).rename('FScore'))#.addBands(prob)


# First step
# create a wraping function
def wrap_addFScore(feature_collection, DOI):
    # define function to process each object
    
    def do_buffer(image):
        image = add_cloud_bands(image)
        # all steps must be able running in the server-side
        #Subtract to get difference, take absolute, then negate (because qualityMosaic only does ascending), then convert to days
        deltaDays = DOI.millis().subtract(image.date().millis()).abs().divide(ee.Number(8.64e+7)).toInt()
        image = image.addBands(ee.Image(deltaDays).rename('delta').toInt())
        prob = image.select('prob').divide(100)
        # image.addBands(prob).rename('prob')
        
        x = ee.Image(deltaDays.add(1).log10()).add(1).multiply(ee.Image(K).pow(prob.add(ee.Number(OFFSET)))).multiply(ee.Image(MULT))
        return image.addBands(x.multiply(ee.Number(-1)).rename('FScore'))
    return feature_collection.map(lambda feat: do_buffer(feat))
