import ee
import geemap
import numpy as np
import pandas as pd
import seaborn as sns

from geemap import ml
from sklearn import ensemble
import quality_mosaic as qm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


keeper_columns = [*qm.FC_columns, 'ID', 'keep']

#add 'Date' to keeper_columns
resample_columns = keeper_columns.copy()
resample_columns.append('Date')
resample_columns.append('Class')

plotting_columns = ['ID', 'Class', 'Latitude','Longitude']

palette = ['008080','f3ff4a','c71585','c0c0c0', '2E86C1','8c411d','00854d','551a4d']

class_labels = ['sand', 'gravel','whitewater','water','greenveg', 'bare', 'cobble']

legend_dict = {
    'fine': '008080', 
    'sand': 'f3ff4a', 
    'gravel': 'c71585',
    'whitewater':'c0c0c0',
    'water': '2E86C1',
    'bare': '8c411d', 
    'greenveg': '00854d',
    'cobble': '551a4d'
}


classy_vizParams = {"min": 0, "max": len(legend_dict)-1, "palette": palette}

random_state = 13

def read_gt(filename, sheets=None, keep_columns=keeper_columns):
    df = pd.read_excel(filename, sheet_name=sheets)
    df = pd.concat(df, ignore_index=True)

    #set the type of 'keep' columns to boolean
    df['keep'] = df['keep'].astype(bool)
    #check if the column 'ID' is unique
    assert(df['ID'].is_unique)

    labels = df[keep_columns]
    #filter out values from labels where the keep column is False and class_code is 99
    labels = labels[(labels['keep'] == True) & (labels['class_code'] != 99) ].reset_index(drop=True)

    #check if there are any null values
    labels.isnull().sum()

    #make sure that there are no nans anywhere
    assert(labels.isnull().sum().sum() == 0)

    #drop any rows that have a class_code of 0
    labels = labels[labels['class_code'] != 0].reset_index(drop=True)

    return labels

def split(filename, sheets=None, test_split=0.3, type='sklearn'):

    labels = read_gt(filename, sheets)
    

    if type == 'sklearn':
        label = 'class_code'
        features = qm.OBIA_BANDS

        # get the features and labels into separate variables
        X = labels[qm.FC_columns] #df[features]
        y = labels[label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
        #drop columns Latitude, Longitude and class_code from X_train
        # X_train = X_train.drop(['Latitude', 'Longitude', 'class_code'], axis=1)
        #Lets keep those columns in X_test

        print("Train test split: ", y_train.shape[0], ":", y_test.shape[0])

        return X_train, X_test, y_train, y_test

def sklearn_rf_to_gee(rf):
    trees = ml.rf_to_strings(rf, qm.OBIA_BANDS)
    gee_classifier = ml.strings_to_classifier(trees)

    return gee_classifier, trees


def train_classifier(df, type='sklearn', output_type=None, n_estimators=100, max_depth=None, test_split=0., split_only_classes_of_int=False, show_label_dist=False, **kwargs):
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



    if type == 'sklearn':

        label = 'class_code'
        features = qm.OBIA_BANDS

        # get the features and labels into separate variables
        X = df[qm.FC_columns] #df[features]
        y = df[label]

        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
            #drop columns Latitude, Longitude and class_code from X_train
            X_train = X_train.drop(['Latitude', 'Longitude', 'class_code'], axis=1)
            #Lets keep those columns in X_test

            print("Train test split: ", y_train.shape[0], ":", y_test.shape[0])

            #concatenate X_train and X_test by creating a new column called type and set it to train or test
            # X_train2, X_test2 = X_train.copy(), X_test.copy()
            # X_train2['type'] = 'train'
            # X_test2['type'] = 'test'
            # #append X_test2 to X_train2
            # labels = pd.concat([X_train2, X_test2])
            # print(len(labels['type']=='train'))
            # display(labels)

            # display(labels2)
            #Using seaborn to plot the distribution of classes in the training and test set in the same plot
            if show_label_dist:
                sns.set(style="darkgrid")
                sns.countplot(x="class_code", data=X_test)
                #set the title of the plot
                plt.title('Distribution of classes in the Test set')
        else:
            X_train = X
            y_train = y
            if show_label_dist:
                sns.set(style="darkgrid")
                sns.countplot(x="class_code", data=X_train, hue="type")
                #set the title of the plot
                plt.title('Distribution of classes in the Train set')

        print(max_depth)
        rf = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs).fit(X_train, y_train)
        #Now convert to GEE classifier
        trees = ml.rf_to_strings(rf, features)
        assert(len(trees) == n_estimators)
        classifier = ml.strings_to_classifier(trees)
        if output_type == 'Prob':
            classifier = classifier.setOutputMode('MULTIPROBABILITY')
        return classifier, trees, (X_test, y_test)
        
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

    

def load_train_rf(filename, sheets=None, show_label_dist=True, rf_type='gee' ,num_trees=100, test_split=0., split_only_classes_of_int=False, **kwargs):
    """
    filename: str, path to the excel file that has the dataset
    sheets: list, list of sheets in the excel file to be used
            if None, all sheets will be concatenated
    show_label_dist: bool, if True, show the distribution of labels
    rf_type: str, type of random forest to train. Valid values are 'gee' and 'sklearn'
    num_trees: int, number of trees in the forest
    test_split: float, fraction of data to be used for testing
            : if zero, no splitting will be done
    split_only_classes_of_int: bool, if True, only the classes of interest will be split

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



    #Now train a model based on this dataset
    if rf_type == 'gee':
        return train_classifier(labels, type='gee', n_estimators=num_trees, show_label_dist=show_label_dist, **kwargs)
        # RF_multi, trees_multi = train_classifier(labels, type='gee', n_estimators=num_trees, **kwargs, output_type='Prob')
    elif rf_type == 'sklearn':
        return train_classifier(labels, type='sklearn', n_estimators=num_trees, 
                                test_split=test_split, split_only_classes_of_int=split_only_classes_of_int,
                                show_label_dist=show_label_dist,
                                **kwargs) 
    else:
        raise Exception('Invalid RF type: {}'.format(rf_type))
    


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


def get_rf_from_gee(modelname:str):
    ### Specify Asset ID
    asset_id = f"projects/gee-sand/assets/{modelname}"

    #### Load classifier
    rf_fc = ee.FeatureCollection(asset_id)

    # convert it to a classifier, very similar to the `ml.trees_to_classifier` function
    new_rf = ml.fc_to_classifier(rf_fc)
    
    return new_rf

def get_confusion_matrix(classified):
    classified = classified.remap(
      [0, 1, 2, 3, 4, 5, 6, 7], 
      [0, 1, 2, 0, 0, 0, 0, 0], 
        'classification'
    )

    classified = classified.remap(
      [0, 1, 2, 3, 4, 5, 6, 7], 
      [0, 1, 2, 0, 0, 0, 0, 0], 
        'class_code'
    )

    test_accuracy = classified.errorMatrix('class_code', 'classification')
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

def overall_accuracy(mat:np.array):
    """
    calculates accuracy from a confusion matrix
    """
    return np.sum(np.diagonal(mat))/np.sum(mat)

def sg_accuracy(mat:np.array):
    """
    calculates accuracy from a confusion matrix
    """
    return np.sum(np.diagonal(mat[0:2]))/np.sum(mat[0:2])

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
    panela = np.stack([class_labels, precision(mat), recall(mat), f1_score(mat) , np.sum(mat, axis = 1)], axis= 1)
    panela = pd.DataFrame(panela)
    panela.columns = ['class', 'precision', 'recall', 'f1-score', 'support']
    panela = panela.round(decimals=2)
    print(panela.to_markdown(index=False))
    return panela
    
def remap_cm(cm, remapped_class_labels = ['sand', 'gravel','other']):
    #make a copy of cm
    remapped_cm = cm.copy()

    #Let's do the rows first
    for i in range(0, 2):
        remapped_cm[i,2] = np.sum(cm[i,2:])
    #columns
    for j in range(0, 2):
        remapped_cm[2,j] = np.sum(cm[2:, j])

    #diagonal
    remapped_cm[2,2] = np.sum(np.diag(cm[2:,2:]))

    #keep only the first 3 rows and columns
    remapped_cm = remapped_cm[0:3, 0:3]
    display(pd.DataFrame(remapped_cm))
    display(round(metrics_table(remapped_cm, remapped_class_labels)))

    return remapped_cm


def display_cm(cm, class_labels=class_labels):


    disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
    disp.plot()
    #remove all grid lines
    plt.grid(False)

    #use class_labels for the x and y ticks
    disp.ax_.set_xticklabels(class_labels, rotation=45, ha='right')
    disp.ax_.set_yticklabels(class_labels, rotation=0, ha='right')
    #set the palette to magma
    disp.im_.set_cmap('magma')
    #make the colorbar 50% smaller

    plt.show()


    
def apply_snic(image, roi=None, size_segmentation=10, compactness = 0,  connectivity = 8, neighborhoodSize = 256, Map=None):
    # Segmentation using a SNIC approach with superpixels.
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
    # snic = snic.reproject (crs = snic.projection(), scale=10)

    if Map is not None:                           
        vizParamsSNIC = {'bands': ['B4_mean','B3_mean','B11_mean'], 'min': 0, 'max': 3000}
        Map.addLayer(snic, vizParamsSNIC,'SNIC', True)
        #To visualize snic result:
        Map.addLayer(snic.randomVisualizer(), None, 'Clusters', True)

    #The next step generates a list of band names from the snic image, but without "clusters"
    #since we don't need to use pixel values of their cluster IDs as a basis for class mapping:
    predictionBands = snic.bandNames().remove("clusters")
    return snic.select(predictionBands)