import pandas as pd
#define function to split catagores and create columns for each catagory 3 and 4 are covered here
def series_str_split(column):
    '''  
     1. take each row and split text into a list
     2. then iterate over list while spliting key and value using '-'
     3. store key value into dictionary (if statment handling first iteration then once keys exist then append value lists) 
     4. convert to DF
    '''
    value_dict = {}
    for i in range(len(column)):
        the_list = column.iloc[i].split(';')
        for j in the_list:
            key, value = j.split('-')
            if key in value_dict:
                value_dict[key] += [int(value)]
            else:
                value_dict[key] = [int(value)]
    return pd.DataFrame(value_dict)