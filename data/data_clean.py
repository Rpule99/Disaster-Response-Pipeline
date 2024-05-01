import pandas as pd

#remove Duplicates
def Duplicate_clean(df):
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    else:
        pass
    return df
