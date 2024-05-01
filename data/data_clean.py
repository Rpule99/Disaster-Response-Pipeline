import pandas as pd

#remove Duplicates
def Duplicate_clean(df):
    '''  
     1. removed duplicated rows and returns cleaned DF
    '''
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    else:
        pass
    return df
