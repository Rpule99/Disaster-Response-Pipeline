import sys
import pandas as pd
import data_wrangling as dw
import data_clean as dc
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    #load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = pd.merge(categories , messages, on='id')
    #expand
    new_catagories = dw.series_str_split(df['categories'])
    #remove the old
    df.drop(columns=['categories'], inplace=True)
    #return concatenated df
    return pd.concat([df, new_catagories], axis=1)



def clean_data(df):
    return dc.Duplicate_clean(df)


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Message', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()