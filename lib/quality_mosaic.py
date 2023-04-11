import ee
import geemap
import numpy as np
from IPython.display import clear_output

K = 1000
OFFSET = -0.5
MULT = 100
CLOUD_FILTER = 10
CLD_PRB_THRESH = 20

USEFUL_BANDS = ["B2","B3","B4","B8","B8A","B11","B12","VV","VH","mTGSI","BSI","NDWI"]

#Viz params
visParamsRGB = {"min": 0, "max": 4000, "bands": ["B4", "B3", "B2"]}
visParamsVV  = {"min": -30, "max": 0, "bands": ["VV"]}
visParamsFScore  = {"min": -50, "max": 0, "bands": ["FScore"]}
green_blue_yellow_orange = ['#008080','#0039e6','#FFFF31','#f85d31']
visParamsMTGSI  = {'min':-0.5, 'max':0.25, 'palette':green_blue_yellow_orange, 'bands':['mTGSI']}


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

def get_s1_s2(roi, date, max_search_window_months:int=6, median_days:int=6,mosaic_method='qm'):
    """
    roi: ee.Geometry type polygon or point
    date: string 'YYYY-MM-DD', e.g. '2019-01-01'
    max_search_window_months: int, max search window in months
    median_days: int, number of days to median filtered S1/S2 images
    mosaic_method: string, either 'qm' or 'median'
        if 'qm', quality mosaic will be used to mosaic S2 images, and max_search_window_months is used
        if 'median', median will be used to mosaic S2 images, and median_days is used

    return:
    s1: ee.Image type
    """
    DOI, start_date_str, end_date_str = get_DOI(date, max_search_window_months)

    s2_cld_col = get_s2_sr_cld_col(roi, start_date_str, end_date_str)

    if (mosaic_method == 'qm'):
        witFScore = wrap_addFScore(s2_cld_col, DOI)
        qm_s2 = witFScore.qualityMosaic('FScore')
    elif (mosaic_method == 'median'):
        qm_s2 = s2_cld_col.median()
    else:
        raise ValueError('mosaic_method must be either qm or median')
    
    s1 = get_s1_median(roi, DOI, start_date_str, end_date_str, median_days)

    #combine s1 and s2
    s1_s2 = qm_s2.addBands(s1)

    #Add VI Bands
    s1_s2 = addNDWI(addBSI(add_mTGSI(s1_s2)))

    #subset only useful bands
    s1_s2 = s1_s2.select(USEFUL_BANDS)

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


def setup_marker_map(Map, s1_s2, lat, lon, sample, id):
    Map.centerObject(sample, 18)
    Map.add_basemap('SATELLITE')
    Map.addLayer(s1_s2, visParamsVV, 'S1', False)
    Map.addLayer(s1_s2, visParamsMTGSI , 'mTGSI')
    Map.addLayer(s1_s2, visParamsRGB, 'S2')
    Map.addLayer(sample,
                {'color':'green', 'opacity':0.5},
                name='buffer')
    name_marker = 'sample_' + str(id)
    Map.add_marker([lat, lon],tooltip=name_marker, name=name_marker, draggable=False)


# Map viz
def setup_split_map(s1_s2, lat, lon, sample):
    
    Map = geemap.Map()
    Map.centerObject(sample, 18) #VERY IMPORTANT: Split map will not render at zoom > 18
    Map.add_basemap('SATELLITE')
    Map.addLayer(s1_s2, visParamsVV, 's1', False)
    
    left_layer = geemap.ee_tile_layer(s1_s2, visParamsRGB, 'S2')
    right_layer = geemap.ee_tile_layer(s1_s2, visParamsMTGSI , 'mTGSI')
    Map.split_map(left_layer, right_layer)
    
    
    #Add the buffer around sampled point
    Map.addLayer(sample,
                {'color':'green', 'opacity':0.5},
                name='buffer')

    Map.add_marker([lat, lon],tooltip='sample', name='sample', draggable=False)
    Map.addLayerControl()
    return Map


def refresh_maps_one_at_a_time(df, Map, index, sampling_buffer_m=5, max_search_window_months:int=6, median_days:int=6, display_smap=True,mosaic_method='qm'):
    """
    Auto incrementing function
    """
    keep = False
    clear_output()

    obs = df.loc[index]
    
    lat, lon = obs['Latitude'], obs['Longitude']

    print("Index: ", index, " ID: ", obs['ID'], "Class: ", obs['Class'], " Country: ", obs['Country'], " Site: ", obs['Site'])
    Map.remove_drawn_features() #remove the previous markers, if any
    new_sample = None

    point = ee.Geometry.Point([lon, lat])
    sample = point.buffer(sampling_buffer_m).bounds()
    roi = point.buffer(100).bounds()

    date = obs['Date']
    
    s1_s2 = get_s1_s2(roi=roi, date=date, max_search_window_months=max_search_window_months,median_days=median_days, mosaic_method=mosaic_method)
    setup_marker_map(Map, s1_s2, lat, lon, sample, obs['ID'])

    if display_smap:
        SMap = setup_split_map(s1_s2, lat, lon, sample)
        display(SMap)

    return s1_s2, sample

def get_values_one_at_a_time(df, s1_s2, sample, Map, index, sampling_buffer_m=5):
    # global INDEX, sampling_buffer_m
    obs = df.loc[index]

    new_sample = None
    if (Map.draw_last_feature is not None):
        Map.draw_last_feature.geometry().getInfo()['coordinates']
        new_sample = Map.draw_last_feature.buffer(sampling_buffer_m).bounds()
        Map.addLayer(new_sample,
                {'color':'green', 'opacity':0.5},
                name='new_buffer')


    inp = input("[{:} {:}] Enter 'y' to keep this sample: ".format(obs['ID'],  obs['Class']))
    keep = inp == 'y'  
    clear_output()

    
    if (keep):
        if new_sample is not None:
            sample = new_sample.geometry()
            print("New marker accepted")
            df['location_tweaked'].loc[index] = True
        else:
            print("Original marker accepted")

        DN_sample = s1_s2.reduceRegion(**{
                    'reducer': ee.Reducer.mean(),
                    'geometry': sample,
                    'scale': 10,
                    'maxPixels': 1e5
                    }).getInfo()


        print("Kept Observation")
        for b, band in enumerate(USEFUL_BANDS):
            # print(b, band)
            df[band].loc[index] = DN_sample[band]

        df['keep'].loc[index] = True
    else:
        print("Discarded Observation")

    index += 1  
    return df, index


def setup_output_bands(output):
    cols = output.columns.values.tolist()

    for band in USEFUL_BANDS:
        if band not in cols:
            output[band] = np.NaN

    output['keep'] = False
    output['keep'] = False
    output['location_tweaked'] = False
    return output


