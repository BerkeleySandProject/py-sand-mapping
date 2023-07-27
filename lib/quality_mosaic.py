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
OBIA_BANDS = [f"{band}_median" for band in USEFUL_BANDS]
#Bands for creating the feature collection to pass to the RF -> because some geometry is required to create an FC, but the lat &lon will not be used for classification
FC_columns = [*OBIA_BANDS, "Longitude","Latitude","class_code"]

#Viz params
visParamsRGB = {"min": 0, "max": 2500, "bands": ["B4", "B3", "B2"]}
visParamsVV  = {"min": -30, "max": 0, "bands": ["VV"]}
visParamsFScore  = {"min": -50, "max": 0, "bands": ["FScore"]}
green_blue_yellow_orange = ['#008080','#0039e6','#FFFF31','#f85d31']
visParamsMTGSI  = {'min':-0.5, 'max':0.25, 'palette':green_blue_yellow_orange, 'bands':['mTGSI']}

#create a dictonary to store mapping between string class names and numeric class values
class_dict = {'fine': 0, 'sand': 1, 'gravel': 2, 'whitewater':3 , 'water': 4, 'bare': 5, 'greenveg': 6, 'other': 7}

legend_dict = {
    'fine': '008080', 
    'sand': 'f3ff4a', 
    'gravel': 'ffa500',
    'whitewater':'ff00ff',
    'water': '2E86C1',
    'bare/impervious/urban': '8c411d', 
    'greenveg': '00854d',
    'other': '551a4d'
}

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
def get_s1_median(roi, DOI, start_date_str, end_date_str, median_samples=0):
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterBounds(roi)\
        .filterDate(start_date_str, end_date_str)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('resolution_meters', 10))
        # .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
        

    # s1_col = s1_col.map(lambda image: image.set(
    #     'time_diff',  ee.Number(image.get('system:time_start')).subtract(DOI.millis()).abs()))
    s1_col = s1_col.map(lambda img: img.set('time_diff', img.date().difference(DOI, 'day').abs()))

    s1_col = s1_col.sort('time_diff')
    if median_samples > 0:
        s1_col = s1_col.limit(median_samples)
    s1 = s1_col.median()

    return s1

def get_DOI(date:str, max_search_window_months:int):
    DOI = ee.Date(date)
    start_date, end_date = DOI.advance(-max_search_window_months, 'month'), DOI.advance(max_search_window_months, 'month')
    start_date_str, end_date_str = start_date.format('YYYY-MM-dd').getInfo(), end_date.format('YYYY-MM-dd').getInfo()
    print("Search window from {:} to {:}".format(start_date_str, end_date_str))
    return DOI, start_date_str, end_date_str

def get_s1_s2(roi, date, max_search_window_months:int=6, median_samples:int=6,mosaic_method='qm', clip=True):
    """
    roi: ee.Geometry type polygon or point
    date: string 'YYYY-MM-DD', e.g. '2019-01-01'
    max_search_window_months: int, max search window in months
    median_samples: int, number of days to median filtered S1/S2 images
    mosaic_method: string, either 'qm' or 'median'
        if 'qm', quality mosaic will be used to mosaic S2 images, and max_search_window_months is used
        if 'median', median will be used to mosaic S2 images, and median_samples is used

    return:
    s1: ee.Image type
    """
    DOI, start_date_str, end_date_str = get_DOI(date, max_search_window_months)

    s2_cld_col = get_s2_sr_cld_col(roi, start_date_str, end_date_str)

    if (mosaic_method == 'qm'):
        witFScore = wrap_addFScore(s2_cld_col, DOI)
        qm_s2 = witFScore.qualityMosaic('FScore')
    elif (mosaic_method == 'median'):
        s2_cld_col = s2_cld_col.sort('time_diff')
        if median_samples > 0:
            s2_cld_col = s2_cld_col.limit(median_samples)
        qm_s2 = s2_cld_col.median()
    else:
        raise ValueError('mosaic_method must be either qm or median')
    
    s1 = get_s1_median(roi, DOI, start_date_str, end_date_str, median_samples)

    #combine s1 and s2
    s1_s2 = qm_s2.addBands(s1)

    #Add VI Bands
    s1_s2 = addNDWI(addBSI(add_mTGSI(s1_s2)))

    #subset only useful bands
    s1_s2 = s1_s2.select(USEFUL_BANDS)

    if clip:
        s1_s2 = s1_s2.clip(roi)

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


def setup_marker_map(Map, s1_s2, lat, lon, id):
    Map.centerObject(ee.Geometry.Point([lon, lat]), 18)
    Map.add_basemap('SATELLITE')
    Map.addLayer(s1_s2, visParamsVV, 'S1', False)
    Map.addLayer(s1_s2, visParamsMTGSI , 'mTGSI')
    Map.addLayer(s1_s2, visParamsRGB, 'S2')
    name_marker = 'sample_' + str(id)
    Map.add_marker([lat, lon],tooltip=name_marker, name=name_marker, draggable=False)


# Map viz
def setup_split_map(s1_s2, lat, lon):
    
    Map = geemap.Map()
    Map.centerObject(ee.Geometry.Point([lon, lat]), 18) #VERY IMPORTANT: Split map will not render at zoom > 18
    Map.add_basemap('SATELLITE')
    Map.addLayer(s1_s2, visParamsVV, 's1', False)
    
    left_layer = geemap.ee_tile_layer(s1_s2, visParamsRGB, 'S2')
    right_layer = geemap.ee_tile_layer(s1_s2, visParamsMTGSI , 'mTGSI')
    Map.split_map(left_layer, right_layer)

    Map.add_marker([lat, lon],tooltip='sample', name='sample', draggable=False)
    Map.addLayerControl()
    return Map

def setup_output_bands(output, obia=True):
    cols = output.columns.values.tolist()

    if obia:
        BANDS = OBIA_BANDS
    else:
        BANDS = USEFUL_BANDS

    for band in BANDS:
        if band not in cols:
            output[band] = np.NaN

    #Check if the output alredy has the keep and location_tweeked columns
    if 'keep' not in cols:
        output['keep'] = False
        output['location_tweaked'] = False
        output['class_code'] = 99
    return output



def get_s1s2_data(df, Map, index, max_search_window_months:int=6, 
                  median_samples:int=6, display_smap=True, mosaic_method='qm', roi_buffer_m=200, 
                  obia=True, interactive = True):
    """
    Auto incrementing function
    """
    keep = False

    if interactive:
        clear_output()

    df = setup_output_bands(df, obia=obia)

    obs = df.loc[index]
    
    lat, lon = obs['Latitude'], obs['Longitude']

    if Map is not None:
        print("Index: ", index, " ID: ", obs['ID'], "Class: ", obs['Class'], " Site: ", obs['Site'])
        Map.remove_drawn_features() #remove the previous markers, if any


    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(roi_buffer_m).bounds()

    date = obs['Date']
    
    s1_s2 = get_s1_s2(roi=roi, date=date, max_search_window_months=max_search_window_months,median_samples=median_samples, mosaic_method=mosaic_method)
    
    if Map is not None:
        setup_marker_map(Map, s1_s2, lat, lon, obs['ID'])

    if display_smap:
        SMap = setup_split_map(s1_s2, lat, lon)
        display(SMap)

    return s1_s2, point


def get_pixel_values(df, s1_s2, sample_point, Map, index, sampling_buffer_m=5):
    # global INDEX, sampling_buffer_m
    obs = df.loc[index]

    new_sample = None
    if (Map.draw_last_feature is not None):
        Map.draw_last_feature.geometry().getInfo()['coordinates']
        new_sample = Map.draw_last_feature.geometry()
        sample_point_with_buffer = new_sample.buffer(sampling_buffer_m).bounds()
    else:
        sample_point_with_buffer = sample_point.buffer(sampling_buffer_m).bounds()

    Map.addLayer(
        sample_point_with_buffer, {'color':'green', 'opacity':0.5}, name='Buffer'
    )

    inp = input("[{:} {:}] Enter 'y' to keep this sample: ".format(obs['ID'],  obs['Class']))
    keep = inp == 'y'  
    clear_output()

    
    if (keep):
        if new_sample is not None:
            print("New marker accepted")
            df['location_tweaked'].loc[index] = True
        else:
            print("Original marker accepted")

        DN_sample = s1_s2.reduceRegion(**{
                    'reducer': ee.Reducer.mean(),
                    'geometry': sample_point_with_buffer,
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


def get_obia_values(df, s1_s2, sample_point, Map, index, 
                    size_seg_px=10, compactness=0.,display_clusters=False, sample_class=None,
                    interactive=True):
    """
    interactive: True if user is manually adjusting labels, index will auto increment
                : False is it's happenning programmatically, index will not auto increment
    
    """
    # global INDEX, sampling_buffer_m
    obs = df.loc[index]
    keep = False

    if sample_class is None: # user has not overridden class variable
        sample_class = obs['Class']

    is_marker_new = False
    if Map is not None or interactive:
        if (Map.draw_last_feature is not None):
            sample_point = Map.draw_last_feature.geometry()
            is_marker_new = True


    if interactive:
        inp = input("[{:} {:}] Enter 'y' to keep this sample: ".format(obs['ID'],  obs['Class']))
        keep = inp == 'y'  
        clear_output()

    
    if keep or not interactive: #if user wants to keep it, or it's running in non-interactive mode
        if is_marker_new:
            print("New marker accepted")
            df['location_tweaked'].loc[index] = True
            df['Longitude'].loc[index], df['Latitude'].loc[index] = Map.draw_last_feature.geometry().getInfo()['coordinates']

        elif interactive:
            print("Original marker accepted")
            df['location_tweaked'].loc[index] = False

        # We reduce the image to a 200m radius around the sampled point.
        # This makes the remaining computations cheaper.
        s1_s2_clipped = s1_s2.clip(sample_point.buffer(200))

        seeds = ee.Algorithms.Image.Segmentation.seedGrid(size_seg_px) #to get spaced grid notes at a distance specified by segmentation size parameter
        snic = ee.Algorithms.Image.Segmentation.SNIC(
                    image=s1_s2_clipped, #our multi-band image with selected bands 
                    compactness=compactness, #if 0, it allows flexibility in object shape, no need to force compactness
                    connectivity=8, #use all 8 neighboring pixels in a pixel neighborhood
                    neighborhoodSize=256, 
                    seeds=seeds)
        snic = snic.reproject(crs=snic.projection(), scale=10)
    
        s1_s2_with_clusters = s1_s2_clipped.addBands(snic.select('clusters'))

        cluster_medians = s1_s2_with_clusters.reduceConnectedComponents(
                    reducer=ee.Reducer.median(),
                    labelBand='clusters')
        
        # The GEE docs recommend to not reproject(). See: https://developers.google.com/earth-engine/guides/image_objects. But we reproject anyway. Reasons:
        # - If we don't reproject, the reduceConnectedComponents() depends on the map zoom level. If we zoom in, we might exceed maxSize -> larger clusters are masked out.
        # - Earlier, we clipped the image to relatively small size. reduceConnectedComponents() won't be too computationally expensive.
        cluster_medians = cluster_medians.reproject(crs=cluster_medians.projection(), scale=10)

        if display_clusters and Map is not None:
            Map.addLayer(cluster_medians, visParamsRGB, 'Cluster Medians', False)
            Map.addLayer(snic.randomVisualizer(), None, 'Clusters', False)

        # .sample(sample_point) returns a FeatureCollection with 1 Feature. That's we select the only Feature using .first()
        DN_obia_sample = cluster_medians.select(USEFUL_BANDS).sample(sample_point).first()
        DN_obia_sample_dict = DN_obia_sample.toDictionary().getInfo()
        
        for band in USEFUL_BANDS:
            print(band, DN_obia_sample_dict[band])
            df[f"{band}_median"].loc[index] = DN_obia_sample_dict[band]

        df['keep'].loc[index] = True
        df['class_code'].loc[index] = class_dict[sample_class.lower()]

        if interactive:
            print("Kept Observation")

    else:
        df['keep'].loc[index] = False
        df['class_code'].loc[index] = 99
        print("Discarded Observation")

    if not interactive:
        index += 1  
    return df, index


def get_training_sample(df, s1_s2, sample_point, Map, index, sampling_buffer_m=5, 
                        size_seg_px=10, compactness=0.,display_clusters=False, 
                        sample_class=None, obia=True, interactive=True):
    if obia:
        df, index = get_obia_values(df, s1_s2, sample_point, Map, index, 
                                    size_seg_px, compactness,display_clusters, sample_class, interactive)
    else:
        df, index = get_pixel_values(df, s1_s2, sample_point, Map, index, sampling_buffer_m)
    return df, index

