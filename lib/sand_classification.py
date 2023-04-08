## Fineness Modulus based classification
import statsmodels.formula.api as sm
import pandas as pd

def get_FM(std_sieves, sample):
    """
    Pass in a df of samples with 2 columns: d and percentage
    returns a tuple of the R2, sum of squared errors and the Fineness modulus (FM)
    """

    model = sm.ols(formula = 'percentage ~ d ', data = sample).fit() #+ I(d**2)

    #predict the values of y for the values of x in model
    
    pred_df = model.predict(std_sieves)
    #apply a ceieling function to the predicted values to cap them at 100 and a floor at 0
    pred_df = pred_df.apply(lambda x: 100 if x > 100 else x)
    pred_df = pred_df.apply(lambda x: 0 if x < 0 else x)
    #sum up the pred_df
    FM = (100.0 - pred_df).sum()/100.0
    return (model.rsquared, model.bse['d'], FM)

def get_FM_df(sieve_df, std_sieves):
    """
    Pass in a df of samples with columns: d* and percentage
        Note: d* is the diameter of the sieve opening and this has to be a contiguous range of numbers that are monotonically increasing
    returns a tuple of the R2, sum of squared errors and the Fineness modulus (FM)

    """
    r2, bse, fm = [],[],[]
    #iterate through each row of sieve_df
    for i in range(len(sieve_df)):
        sample = sieve_df.iloc[i].copy()
        sample = sample.to_frame()
        sample.columns = ['d']
        sample.index.name = 'percentage'
        sample['percentage'] = sample.index.astype(float)
        sample = sample.reset_index(drop=True) 
        r2_sample, bse_sample, fm_sample, = get_FM(std_sieves, sample)
        r2.append(r2_sample)
        fm.append(fm_sample)
        bse.append(bse_sample)

    #convert the list fm to a column of sieve_df called 'FM'
    sieve_df['FM'] = fm
    sieve_df['r2_fm'] = r2
    sieve_df['stderr_fm'] = bse
    return sieve_df

def assign_FM_classes(df, FM_sand_min =1.71, FM_sand_max=4.0):
    """
    df is all the samples that have a set of columns in the format 'd*' representing the diameter of the sieve opening and 'percentage' representing the percentage passing the sieve
    FM_sand_min is the minimum FM value to be considered sand
    FM_sand_max is the maximum FM value to be considered sand
    """

    #grab only the columns of df_merged that start with 'd' and is followed by a number
    sieve_df = df.filter(regex='^d\d+$')

    #rename the columns of sieve_df to be the number that follows 'd'
    sieve_df.columns = sieve_df.columns.str.replace('d', '')

    #define std sieves
    std_sieves = pd.DataFrame({'d': [10., 4.75, 2.36, 1.18, 0.6, 0.3, 0.15]})

    fm_df = get_FM_df(sieve_df, std_sieves)
    #Filter out bad fits
    fm_df = fm_df[fm_df['r2_fm'] >= 0.8]
    
    #merge df back with the merged_df on the index of df and keep all the columns from df and only the columns 'FM', 'r2' and 'stderr' from sieve_df
    fm_df = df.merge(sieve_df[['FM', 'r2_fm', 'stderr_fm']], left_index=True, right_index=True)
    
    #create a new column of type 'str' in fm_df called 'class' and set it to 'fine' if the FM is less than 1.71 and 'coarse' if the FM is greater than 4 and 'sand' otherwise
    fm_df['FM_class'] = fm_df['FM'].apply(lambda x: 'fine' if x < 1.71 else 'coarse' if x > 4. else 'sand')

    return fm_df


def get_grid_sieves(grid_sieves, sample):
    """
    Pass in a df of samples with 2 columns: d and percentage
    returns a tuple of the R2, sum of squared errors and the Fineness modulus (FM)
    """

    model = sm.ols(formula = 'percentage ~ d ', data = sample).fit() #+ I(d**2)

    #predict the values of y for the values of x in model
    
    pred_df = model.predict(grid_sieves)
    #apply a ceieling function to the predicted values to cap them at 100 and a floor at 0
    pred_df = pred_df.apply(lambda x: 100 if x > 100 else x)
    pred_df = pred_df.apply(lambda x: 0 if x < 0 else x)

    return (model.rsquared, model.bse['d'], pred_df)


def get_grid_df(sieve_df, grid_sieves):

    r2, bse, D75, D4_75, D_075 = [],[],[],[],[]
    #iterate through each row of sieve_df
    for i in range(len(sieve_df)):
        sample = sieve_df.iloc[i].copy()
        sample = sample.to_frame()
        sample.columns = ['d']
        sample.index.name = 'percentage'
        sample['percentage'] = sample.index.astype(float)
        sample = sample.reset_index(drop=True) 
        r2_sample, bse_sample, pred_df = get_grid_sieves(grid_sieves, sample)

        sample_D75, sample_D4_75, sample_D_075 = pred_df.iloc[0], pred_df.iloc[1], pred_df.iloc[2]

        r2.append(r2_sample)
        D75.append(sample_D75)
        D4_75.append(sample_D4_75)
        D_075.append(sample_D_075)
        bse.append(bse_sample)

    #convert the list fm to a column of sieve_df called 'FM's
    sieve_df['r2_grid'] = r2
    sieve_df['stderr_grid'] = bse
    sieve_df['D75'] = D75
    sieve_df['D4_75'] = D4_75
    sieve_df['D_075'] = D_075
    return sieve_df

def assign_grid_classes(df):
    sieve_df = df.filter(regex='^d\d+$')

    #rename the columns of sieve_df to be the number that follows 'd'
    sieve_df.columns = sieve_df.columns.str.replace('d', '')

    grid_sieves = pd.DataFrame({'d': [75., 4.75, 0.075]})

    fm_df = get_grid_df(sieve_df, grid_sieves)
    #Filter out bad fits
    fm_df = fm_df[fm_df['r2_grid'] >= 0.8]
    
    #merge df back with the merged_df on the index of df and keep all the columns from df and only the columns 'FM', 'r2' and 'stderr' from sieve_df
    fm_df = df.merge(sieve_df[['D75','D4_75', 'D_075','r2_grid', 'stderr_grid']], left_index=True, right_index=True)
    
    #create a new column of type 'str' in fm_df called 'class' and set it to 'fine' if the FM is less than 1.71 and 'coarse' if the FM is greater than 4 and 'sand' otherwise
    # fm_df['FM_class'] = fm_df['FM'].apply(lambda x: 'fine' if x < 1.71 else 'coarse' if x > 4. else 'sand')
    # create a new column of type 'str' in fm_df called 'class_grid' and set it to 'sand' if D4_75 is more than 50 and D_075 is less than 15
    # and 'coarse' if D4_75 is less than 50 and D_075 is less than 15
    # and 'fine' otherwise
    fm_df['grid_class'] = fm_df.apply(lambda x: 'sand' if x['D4_75'] >= 50 and x['D_075'] <= 15 \
                                      else 'coarse' if x['D4_75'] < 50 and x['D75'] == 100 and x['D_075'] <= 15\
                                         else 'fine', axis=1)

    return fm_df
