from config import config
import psycopg2
import pandas as pd
import numpy as np
import pandas.io.sql as sqlio


def connect():
    conn=None
    params = config()
    print('connecting to PostGres ..')
    try:
        print('Connecting to PG server')
        conn = psycopg2.connect(**params)
        print('Connected')
        crsr = conn.cursor()
        crsr.execute('SELECT version()')
        # db_version = crsr.fetchone()
        # print(db_version)
        # crsr.close()
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None

    # finally:
    #     if conn is not None:
    #         conn.close()
    #         print("Connection Terminated")

def query(conn,table_name):
    try:
        crsr = conn.cursor()
        print("cursor created")
        # crsr.execute('SELECT * FROM source_file')

        # check if rows are returned
        query = f"SELECT * FROM {table_name}"
        print(conn)
        res  = crsr.execute(query)
        print('res')
        print(res)

        data = crsr.fetchall()
        # print(data)
        # selecting columns in the db
        columns = [desc[0] for desc in crsr.description]
        # print(columns)

        df = pd.DataFrame(data,columns=columns)
        crsr.close()
        return df

    except (Exception, psycopg2.DatabaseError) as error:
        print(f'Query failed: {error}')
        return pd.DataFrame

# def run_query(tablename,conn):
#     dat = pd.read_sql_query(f"SELECT * FROM {tablename}", conn)
#     # conn=None
#     return dat

# def describe_data(df):
#     total_col = df.columns.to_list()
#     print(total_col)
#     print(len(total_col))


# def is_feature_matched(df,ml_features):
#     df_col = df.columns().to_list()
#     if df_col == ml_features:
#         return True
#     else:
#         print("Features mismatch")

def feature_select(df:pd.DataFrame):
    num_col = df.select_dtypes(include=['int','float'])
    cat_col = df.select_dtypes(include=['object'])
    date_col = df.select_dtypes(include=['datetime64'])
    print(f"Pre_processed_features are :\nNumeric col:\n{num_col}\n----\nCategorical col:\n{cat_col}\n----\nDate col:\n{date_col}")

def feature_engineering(df):
    # df['Age'] 
    pass

def scale_data(df):
    pass

def encode_data(df):
    
    pass




if __name__ == "__main__":
    conn = connect()
    if conn is not None:
        table_name='source_file'
        # df1 = run_query(tablename=table_name,conn=conn)
        # print(df1.head())
        data = query(conn,table_name=table_name)
        # print(data)
        # print(df)
        # data1,res = query(conn)
        # print(data1)
        # print(res)
        # print(df)
        # print(df.head())
        # conn.close()
        print('Connection Terminated')
        # print(query(conn))


