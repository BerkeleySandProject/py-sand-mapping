import ee
import geemap
import numpy as np
import pandas as pd
import seaborn as sns

from geemap import ml
from sklearn import ensemble
import quality_mosaic as qm

keeper_columns = ['ID', 'class_code', 'B2_mean', 'B3_mean', 'B4_mean', 'B8_mean',
       'B8A_mean', 'B11_mean', 'B12_mean', 'VV_mean', 'VH_mean', 'mTGSI_mean',
       'BSI_mean', 'NDWI_mean', 'keep','Latitude','Longitude']

plotting_columns = ['ID', 'Class', 'Latitude','Longitude']

palette = ['008080','f3ff4a','c71585','c0c0c0', '2E86C1','8c411d','00854d','551a4d']

legend_dict = {
    'fine': '008080', 
    'sand': 'f3ff4a', 
    'gravel': 'c71585',
    'whitewater':'c0c0c0',
    'water': '2E86C1',
    'bare': '8c411d', 
    'greenveg': '00854d',
    'other': '551a4d'
}

classy_vizParams = {"min": 0, "max": len(legend_dict)-1, "palette": palette}

def train_classifier(df, type='sklearn', output_type=None, n_estimators=100, max_depth=10):
    """
    Train a classifier using the specified type.
    :param df: A pandas dataframe containing the training data. It must contain all the bands of interest, latitude, logitude and class_code.
    :param type: The type of classifier to train. Valid values are 'sklearn' and 'gee'.
    :param n_estimators: The number of trees in the forest
    :param max_depth: The maximum depth of the tree
    Returns: tuple containing:
        A trained GEE classifier.
        trees: string list of trees in the forest so it can be saved later
    """

    random_state = 13

    if type == 'sklearn':
        label = 'class_code'
        features = qm.OBIA_BANDS

        # get the features and labels into separate variables
        X = df[features]
        y = df[label]

        rf = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state).fit(X, y)
        #Now convert to GEE classifier
        trees = ml.rf_to_strings(rf, features)
        assert(len(trees) == n_estimators)
        classifier = ml.strings_to_classifier(trees)
        if output_type == 'Prob':
            classifier = classifier.setOutputMode('MULTIPROBABILITY')

        
    elif type == 'gee':
        # get the features and labels into separate variables
        X = geemap.common.pandas_to_ee(df[qm.FC_columns], 'Longitude', 'Latitude')

        if output_type == 'Prob':
            classifier = ee.Classifier.smileRandomForest(n_estimators, seed=random_state).setOutputMode('MULTIPROBABILITY').train(X, 'class_code', qm.OBIA_BANDS)
        else:
            classifier = ee.Classifier.smileRandomForest(n_estimators, seed=random_state).train(X, 'class_code', qm.OBIA_BANDS)
    
        trees = ee.List(ee.Dictionary(classifier.explain()).get('trees')).getInfo()
    else:
        raise Exception('Invalid classifier type: {}'.format(type))


    return classifier, trees


def train_rf(filename, sheets=None, show_label_dist=True, rf_type='gee' ,num_trees=100, **kwargs):
    """
    filename: str, path to the excel file that has the dataset
    sheets: list, list of sheets in the excel file to be used
            if None, all sheets will be concatenated
    show_label_dist: bool, if True, show the distribution of labels
    rf_type: str, type of random forest to train. Valid values are 'gee' and 'sklearn'

    returns: tuple containing:
        RF: A trained GEE classifier or sklearn classifier
        trees: string list of trees in the forest so it can be saved later
        RF_multi: A trained GEE classifier in MULTIPROBABILITY mode
        trees_multi: string list of trees in the forest so it can be saved later

    """

    df = pd.read_excel(filename, sheet_name=sheets)
    df = pd.concat(df, ignore_index=True)

    #set the type of 'keep' columns to boolean
    df['keep'] = df['keep'].astype(bool)
    #check if the column 'ID' is unique
    assert(df['ID'].is_unique)

    labels = df[keeper_columns]
    labels = labels[labels['keep'] == True].reset_index(drop=True)

    #check if there are any null values
    labels.isnull().sum()

    #make sure that there are no nans anywhere
    assert(labels.isnull().sum().sum() == 0)

    if show_label_dist:
        sns.set(style="darkgrid")
        sns.countplot(x="class_code", data=labels)

    #Now train a model based on this dataset
    if rf_type == 'gee':
        RF, trees             = train_classifier(labels, type='gee', n_estimators=num_trees, **kwargs)
        # RF_multi, trees_multi = train_classifier(labels, type='gee', n_estimators=num_trees, **kwargs, output_type='Prob')
    else:
        RF, trees             = train_classifier(labels, type='sklearn', n_estimators=num_trees, **kwargs)
        ## Cannot do prob output with sklearn trees in GEE
        # RF_multi, trees_multi = train_classifier(labels, type='gee', n_estimators=num_trees, **kwargs, output_type='Prob')

    return RF, trees


def save_trees_gee(trees, asset_name):
    """
    Save the trees to a GEE asset
    :param trees: list of trees
    :param asset_id: GEE asset id
    """
    # convert the list of trees to a string
    asset_id =  f"projects/gee-sand/assets/{asset_name}"
    print(asset_id)
    ml.export_trees_to_fc(trees, asset_id)




def get_confusion_matrix(classified):
    classified = classified.remap(
      [0, 1, 2, 3, 4, 5, 6, 7], 
      [0, 1, 2, 0, 0, 0, 0, 0], 
        'classification'
    )

    classified = classified.remap(
      [0, 1, 2, 3, 4, 5, 6, 7], 
      [0, 1, 2, 0, 0, 0, 0, 0], 
        'classcode'
    )

    test_accuracy = classified.errorMatrix('classcode', 'classification')
    mat = np.array(test_accuracy.getInfo())
    return mat

def support(mat:np.array):
    """
    calculates support from a confusion matrix
    """
    return np.sum(mat, axis = 1)/np.sum(mat)

def accuracy(mat:np.array):
    """
    calculates accuracy from a confusion matrix
    """
    return np.sum(np.diagonal(mat))/np.sum(mat)

def precision(mat:np.array):
    """
    calculates precision (class-wise) from a confusion matrix
    """
    return np.nan_to_num(np.diagonal(mat)/np.sum(mat, axis = 0))
    
def recall(mat:np.array):
    """
    calculates recall (class-wise) from a confusion matrix
    """
    return np.nan_to_num(np.diagonal(mat)/np.sum(mat, axis = 1))

def f1_score(mat:np.array):
    """
    calculates f1-score (class-wise) from a confusion matrix
    """
    prec = precision(mat)
    rec = recall(mat)
    return np.nan_to_num(2 * prec * rec / (prec + rec))

def macro_precision(mat:np.array):
    """
    calculates macro-averaged precision from a confusion matrix
    """
    assert mat.shape[0] > 2
    return np.mean(precision(mat))

def macro_recall(mat:np.array):
    """
    calculates macro-averaged recall from a confusion matrix
    """
    assert mat.shape[0] > 2
    return np.mean(recall(mat))

def macro_f1score(mat:np.array):
    """
    calculates macro-averaged F1 score from a confusion matrix
    """
    assert mat.shape[0] > 2
    return np.mean(f1_score(mat))

def sklearn_metrics_table(mat:np.array, class_labels = None):
    """
    Generates a table with class-wise precion, recall, f1-score. 
    Also has accuracy, and macro-averaged precision/ recall/ f1-score
    """
    
    panela = np.stack([precision(mat), recall(mat), f1_score(mat), np.sum(mat, axis = 1)], axis = 1)
    panelb = np.stack([[np.nan,np.nan  , accuracy(mat), np.sum(mat)], 
                       [macro_precision(mat), macro_recall(mat), macro_f1score(mat), np.sum(mat)]])


    assert panela.shape[1] == panelb.shape[1]
    mtable = pd.DataFrame(np.append(panela, panelb).reshape(panela.shape[0] + panelb.shape[0], panela.shape[1]))
    
    
    if class_labels:
        assert panela.shape[0] == len(class_labels)
        mtable.index = class_labels + ['accuracy', 'macro avg']
    else:
        mtable.index = ['','', ''] + ['accuracy', 'macro avg']
        
    mtable.columns = ['precision', 'recall', 'f1-score', 'support']
    
    return mtable.fillna('')

def metrics_table(mat:np.array, class_labels:list):
    """
    Returns a dataframe with class-wise precision, recall and f1-score for a given confusion matrix
    """
    panela = np.stack([class_labels, precision(mat), recall(mat), f1_score(mat), np.sum(mat, axis = 1)], axis= 1)
    panela = pd.DataFrame(panela)
    panela.columns = ['class', 'precision', 'recall', 'f1-score', 'support']
    return panela
    
    
    
def apply_snic(image, roi=None, size_segmentation=10, compactness = 0,  connectivity = 8, neighborhoodSize = 256, Map=None):
    # Segmentation using a SNIC approach based on the dataset previosly generated
    seeds = ee.Algorithms.Image.Segmentation.seedGrid(size_segmentation); #to get spaced grid notes at a distance specified by segmentation size parameter
    
    if roi is not None:
        image = image.clip(roi)
    
    snic = ee.Algorithms.Image.Segmentation.SNIC(
                                image = image, #our multi-band image with selected bands same as for pixel-based
                                compactness = compactness,  #allow flexibility in object shape, no need to force compactness
                                connectivity = connectivity, #use all 8 neighboring pixels in a pixel neighborhood
                                neighborhoodSize = neighborhoodSize,
                                seeds = seeds #use the seeds we generated above
                                )
    if Map is not None:                           
        vizParamsSNIC = {'bands': ['B4_mean','B3_mean','B11_mean'], 'min': 0, 'max': 3000}
        Map.addLayer(snic, vizParamsSNIC,'SNIC', True)
        #To visualize snic result:
        Map.addLayer(snic.randomVisualizer(), None, 'Clusters', True)

    #The next step generates a list of band names from the snic image, but without "clusters"
    #since we don't need to use pixel values of their cluster IDs as a basis for class mapping:
    predictionBands = snic.bandNames().remove("clusters")
    return snic.select(predictionBands)