from sqlalchemy import create_engine

def save_data_to_db(df, table_name):
    '''  
     1. create instance of engine
     2. save df to database, replace table if already exists
     3. return error message if data fails to save
    '''
    try:
        engine = create_engine('sqlite:///data/DisasterResponse.db')
        df.to_sql(table_name, engine, if_exists='replace',index=False)
        return print('Dataframe saved to DB')
    except Exception as inst:
        return print('failed to data to table, error: ', inst)

    
