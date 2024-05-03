import pandas as pd

#remove Duplicates
def Duplicate_clean(df):
    '''  
     1. removed duplicated rows and returns cleaned DF
     2. remove where related = 2
    '''
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    else:
        pass
    df = df[df['related'] != 2]
    return df
