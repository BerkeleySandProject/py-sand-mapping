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


## S1 Image
def get_s1_median(roi, DOI, start_date_str, end_date_str, median_days=0):
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterBounds(roi)\
        .filterDate(start_date_str, end_date_str)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
        .filter(ee.Filter.eq('resolution_meters', 10))

    # s1_col = s1_col.map(lambda image: image.set(
    #     'time_diff',  ee.Number(image.get('system:time_start')).subtract(DOI.millis()).abs()))
    s1_col = s1_col.map(lambda img: img.set('time_diff', img.date().difference(DOI, 'day').abs()))

    s1_col = s1_col.sort('time_diff')
    if median_days > 0:
        s1_col = s1_col.limit(median_days)
    s1 = s1_col.median()

    return s1

def get_DOI(date:str, max_search_window_months:int):
    DOI = ee.Date(date)
    start_date, end_date = DOI.advance(-max_search_window_months, 'month'), DOI.advance(max_search_window_months, 'month')
    start_date_str, end_date_str = start_date.format('YYYY-MM-dd').getInfo(), end_date.format('YYYY-MM-dd').getInfo()
    print("Search window from {:} to {:}".format(start_date_str, end_date_str))
    return DOI, start_date_str, end_date_str

def get_s1_s2(roi, date, max_search_window_months:int=6, s1_median_days=0):
    """
    roi: ee.Geometry type polygon or point
    date: string 'YYYY-MM-DD', e.g. '2019-01-01'
    max_search_window_months: int, max search window in months
    s1_median_days: int, number of days to median filtered S1 images

    return:
    s1: ee.Image type
    """
    DOI, start_date_str, end_date_str = get_DOI(date, max_search_window_months)

    s2_cld_col = get_s2_sr_cld_col(roi, start_date_str, end_date_str)
    witFScore = wrap_addFScore(s2_cld_col, DOI)
    qm_s2 = witFScore.qualityMosaic('FScore')

    s1 = get_s1_median(roi, DOI, start_date_str, end_date_str, s1_median_days)

    #combine s1 and s2
    s1_s2 = qm_s2.addBands(s1)


    return s1_s2

# Add indices for S2 images
def add_mTGSI(image):
    mTGSI = image.expression('(R - B + SWIR2 - NIR) / (R + G + B + SWIR2 + NIR)', 
                             { 'R': image.select('B4'),
                              'G': image.select('B3'),
                              'B': image.select('B2'),
                              'NIR': image.select('B8'),
                              'SWIR2': image.select('B12')
                              # 'SWIR1': image.select('B11')
                            }).rename('mTGSI')
    return image.addBands(mTGSI)

def addEVI(image):
    EVI = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
    'NIR' : image.select('B8'),
    'RED' : image.select('B4'),
    'BLUE': image.select('B2')}).rename('EVI')
    
    return image.addBands(EVI)

def addBSI(image):
    BSI = image.expression('((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE))', {
    'NIR' : image.select('B8'),
    'RED' : image.select('B4'),
    'SWIR1': image.select('B11'),
    'BLUE': image.select('B2')
    }).rename('BSI')
    
    return image.addBands(BSI)

def addNDWI(image):
    NDWI = image.expression('(GREEN - NIR) / (GREEN + NIR)', {
    'NIR' : image.select('B8'),
    'GREEN' : image.select('B3')
    }).rename('NDWI')
    
    return image.addBands(NDWI)

