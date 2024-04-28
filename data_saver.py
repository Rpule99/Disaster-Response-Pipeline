from sqlalchemy import create_engine

def save_data_to_db(df, table_name):
    try:
        engine = create_engine('sqlite:///DisasterResponse.db')
        df.to_sql(table_name, engine, if_exists='replace',index=False)
        return print('Dataframe saved to DB')
    except Exception as inst:
        return print('failed to data to table, error: ', inst)

    
