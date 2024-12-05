import pandas as pd 
import numpy as np 

from logging import Logger
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn import pipeline
from test_conn import query ,connect
# import joblib
import xgboost as xgb

# table_name = "source_file"
# data = query(conn,table_name=table_name)
# data = query()


conn = connect()
if conn is not None:
    table_name='source_file'
        # df1 = run_query(tablename=table_name,conn=conn)
        # print(df1.head())
    data = query(conn,table_name=table_name)

def describe_data(df:pd.DataFrame)->pd.DataFrame:
    total_col = df.columns.to_list()
    # print(total_col)
    print(len(total_col))


def is_feature_match(df,ml_features):
    df_col = df.columns.to_list()
    if df_col == ml_features:
        print("Columns matching Successful")
        return True
    else:
        print("Features mismatch")

def feature_select(df:pd.DataFrame) -> pd.DataFrame:
    num_col = df.select_dtypes(include=['int','float'])
    cat_col = df.select_dtypes(include=['object'])
    date_col = df.select_dtypes(include=['datetime64'])
    print(f"Pre_processed_features are :\nNumeric col:\n{num_col}\n----\nCategorical col:\n{cat_col}\n----\nDate col:\n{date_col}")

ml_features = [
    'Last Funding Amount (in USD)', 'Number of Funding Rounds',
    'Last Equity Funding Amount (in USD)', 'Total Funding Amount (in USD)',
    'Number of Articles', 'Total Products Active', 'Number of Investors',
    'Number of Founders', 'Trend Score (30 Days)', 'Trend Score (90 Days)',
    'Monthly Rank Growth', 'Global Traffic Rank', 'emp__10001+',
    'emp__1001-5000', 'emp__101-250', 'emp__11-50', 'emp__251-500',
    'emp__5001-10000', 'emp__501-1000', 'emp__51-100', 'Age_of_startup',
    'time_since_last_funding', 'acquired', 'Debt Financing', 'Post-IPO Equity',
    'Pre-Seed', 'Private Equity', 'Seed', 'Series A', 'Series B', 'Series C',
    'Series F', 'Venture - Series Unknown']

def drop_col(df:pd.DataFrame)->pd.DataFrame:
    col_to_drop = ["Organization URL",'Founded Date Precision','Closed Date Precision','Company Type', 'Investor Type','Last Funding Amount', 'Last Funding Amount Currency',
               'Last Equity Funding Amount', 'Last Equity Funding Amount Currency','Total Equity Funding Amount', 'Total Equity Funding Amount Currency',
               'Total Funding Amount','Total Funding Amount Currency','Diversity Spotlight','Number of Sub-Orgs', 'Stage', 'Most Recent Valuation Range',
               'Date of Most Recent Valuation','Acquired by URL', 'Transaction Name','Number of Exits','Exit Date', 'Exit Date Precision',
               'Description', 'Website', 'Twitter', 'Facebook','Contact Email', 'Phone Number', 'Full Description','Closed Date']
    df = df.drop(columns=col_to_drop,errors='ignore')

def year_creation(df:pd.DataFrame):
    df['Founded Year'] = pd.to_datetime(df['Founded Date']).dt.year
    df['Last Funding Year'] = pd.to_datetime(df['Last Funding Date']).dt.year
    return df 


def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    # 
    
    curr_year = datetime.now().year
    
    # df['Founded Year'] = pd.to_datetime(df['Founded Year'], format='%Y').dt.year
    # df['Last Funding Year'] = pd.to_datetime(df['Last Funding Date'], format='%Y').dt.year
    df['Age_of_startup'] = curr_year - df['Founded Year']
    df['time_since_last_funding'] = curr_year - df['Last Funding Year']



def remove_Duplicates(df:pd.DataFrame):
    # Remove duplicates
    df = df.drop_duplicates(subset=['Organization Name'])
    


def transform_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to transform the data into a suitable format for the model.
    """
    # Removing Duplicates 
    try:
        print(df['Global Traffic Rank'].dtype)
        # df['Global Traffic Rank'] = df['Global Traffic Rank'].str.replace(',','').astype(float)
        # df['Monthly Rank Growth'] = (df['Monthly Rank Growth'].str.replace(',','').str.rstrip('%')).astype(float) / 100

        # df['Global Traffic Rank'] = df['Global Traffic Rank'].str.replace(',', '', regex=False)    
            
        # df['Global Traffic Rank'] = df['Global Traffic Rank'].fillna(0)

        # # df['Global Traffic Rank'] = pd.to_numeric(df['Global Traffic Rank'], errors='coerce')
        # df['Global Traffic Rank'] = df['Global Traffic Rank'].astype('float64')
        
        print("Global part done ")
        # Cleaning Monthly Rank Gro
            # df['Monthly Rank Growth'] = df['Monthly Rank Growth'].str.strip().str.rstrip('%')  # Handle string 'nan'
        print(df['Monthly Rank Growth'].dtype)
    # Remove unwanted characters using regex
        df['Monthly Rank Growth'] = df['Monthly Rank Growth'].str.replace(r'[^\d\.-]', '', regex=True)  # Keep only numbers, dot, and minus)
        # df['Monthly Rank Growth'] = df['Monthly Rank Growth'].astype('float')
    # Convert to numeric
        df['Monthly Rank Growth'] = pd.to_numeric(df['Monthly Rank Growth'], errors='coerce')
            
                # Convert to float and normalize percentage
        # print(df['Monthly Rank Growth'].dtype)
        # print(df['Monthly Rank Growth'].head())
        # df['Monthly Rank Growth'] = df['Monthly Rank Growth'].fillna(0)
        # df['Monthly Rank Growth'] = pd.to_numeric(df['Monthly Rank Growth'], errors='coerce')
        df['Monthly Rank Growth'] = df['Monthly Rank Growth']/ 100
            # Transforming date columns 
        print(df['Monthly Rank Growth'].dtype)
        # print(df['Monthly Rank Growth'].head())
        # Extract year from date column
        print("exiting function")

        # Aquired_by -> acquired
        df['acquired'] = df['Acquired by'].notna().astype(int)

    except Exception as e:
        print(f"Error in transforming data: {e}")
    return df


    

def scale_data(df:pd.DataFrame)-> pd.DataFrame:
    scaler = RobustScaler()
    col_to_scale = ['Total Funding Amount (in USD)','Last Funding Amount (in USD)','Last Equity Funding Amount (in USD)','Global Traffic Rank','Monthly Rank Growth']
    df[col_to_scale] = scaler.fit_transform(df[col_to_scale])
    
    return df

    

# def encode_emp(df:pd.DataFrame)->pd.DataFrame:
    # Last Funding Type a d NUmber of emp
    # col_to_encode = ['Last Funding Type','Number of Employees']
    # # last funding type
    # try:
    #     emp_dummies = pd.get_dummies(df,columns=['Number of Employees'],drop_first=True,prefix='emp_')
    # # emp_dummies = emp_dummies.astype(int)
    # except ValueError as error:
    #     print(f"Error in encoding data: {error}")
    # df = pd.concat([df,emp_dummies],axis=1)
    # # print(df.columns)
    # print("Created dummy columns:", emp_dummies.columns.tolist())

    # return df
def encode_emp(df: pd.DataFrame) -> pd.DataFrame:
    # Encoding 'Number of Employees'
    try:
        # Generate dummy variables for 'Number of Employees' with prefix
        emp_dummies = pd.get_dummies(df['Number of Employees'], drop_first=True, prefix='emp_')
        
        # Print only the newly created dummy columns
        print("Created dummy columns:")
        encoded_emp_list =  emp_dummies.columns.tolist()
        # Concatenate the dummy variables with the original DataFrame
        df = pd.concat([df, emp_dummies], axis=1)
        print("Concatanation is done")
        return df,encoded_emp_list
    except ValueError as error:
        print(f"Error in encoding data: {error}")

        return df,[]


# global encoded_emp_list


def encode_fund_type(df:pd.DataFrame)->pd.DataFrame:
    last_funding_dummies = pd.get_dummies(df['Last Funding Type'],drop_first=True)
    print(last_funding_dummies.columns.tolist())
    last_funding_dummies = last_funding_dummies.astype(int)
  
    df = pd.concat([df,last_funding_dummies],axis=1)
    
    return df

def encode_emp_to_numeric(df:pd.DataFrame,encoded_emp_list:list) -> pd.DataFrame: # fix later for according to the encoding
    # trans_emp = [ 'emp_1001-5000', 'emp_101-250',
    #    'emp_11-50', 'emp_251-500', 'emp_5001-10000', 'emp_501-1000',
    #    'emp_51-100']
    # trans_emp = ['emp_1-10', 'emp_1001-5000', 'emp_101-250', 'emp_11-50', 'emp_251-500', 'emp_5001-10000', 'emp_501-1000', 'emp_51-100']
    # global encoded_emp_list
    
    # convert employees data
    df[encoded_emp_list] = df[encoded_emp_list].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    return df

# print(data.head(5))

# def last_col_drop(df:pd.DataFrame)->pd.DataFrame:
#     pass
#     # for matching with ml_features


# def split_features_target(df:pd.DataFrame)->pd.DataFrame:
#     pass


# Last preprocessing 
def last_preprocessing(df:pd.DataFrame):
    # if df.columns.to_list() != ml_features:
    #     print("Error: The columns of the DataFrame do not match the ml_features list.")
    # # df =df [ml_features]
    
    # return df
    missing_features = [feature for feature in ml_features if feature not in df.columns]
    
    if missing_features:
        print(f"Missing features: {missing_features}")
        # Add missing columns with value 0
        for feature in missing_features:
            df[feature] = 0
        print("Missing features added with default value 0.")
    
    # Optionally, you can re-order the columns to match the order of ml_features
    # df = df[ml_features]  # This ensures the DataFrame has the exact columns as in ml_features
    
    return df


# def sequence_match(data:pd.DataFrame,ml_features):
#     one_data = data.copy()
#     data = da
# def feature_validation(df:pd.DataFrame,model_features):
#      data_features = df.iloc[:, 1:]
#     # assert list(data_features.index) == list(model_features.index), "Feature mismatch between data and model"
#     pass

def ensure_float(df:pd.DataFrame):
    df[-1,1,:]

def validate_sequence(df:pd.DataFrame,ml_feautres):
    data_features = df.columns[1:]  # Skip the first column
    
    # Validate that the sequence matches
    assert list(data_features) == list(ml_features), \
        f"Feature sequence mismatch: Data features {list(data_features)} != Model features {list(ml_features)}"

# def get_model(file_path):
#     pass
    # file_path = "path to model"
    # model = joblib.load(file_path)
    # # with open(file_path,'rb') as f:
    # #     model = f.read()
    # print("Model succesfully acquired")
    # return model
    # try:
    #     with open(file_path, 'rb') as f:
    #         model = joblib.load(f)
    #     print("Model successfully acquired.")
    #     return model
    # except Exception as e:
    #     print(f"Error loading model: {e}")
    #     return None
    

def get_model(file_path):
    try:
        # Load the model using XGBoost's load_model method
        model = xgb.Booster()
        model.load_model(file_path)  # This will load the model from the XGBoost's native format
        print("Model successfully acquired.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Example usage
# model = get_model("latest_xgboost_model.pkl")



# def predict_score(df:pd.DataFrame,model,ml_features):
#     df['Scorecard_1'] = model.predict(df[ml_features])
#     df.to_csv("Testing_Score.csv",index=False)
#     # q = 
#     return df

def predict_score(df: pd.DataFrame, model, ml_features):

    df[ml_features] = df[ml_features].apply(pd.to_numeric, errors='coerce')
    # Convert the DataFrame to a DMatrix
    dmatrix = xgb.DMatrix(df[ml_features],enable_categorical=True)

    # Use the model to make predictions
    df['Scorecard_1'] = model.predict(dmatrix)
    df['Scorecard_1'] = df['Scorecard_1'] * 10
    # df['Scorecard_1'] = df["Scorecard_1"] +1

    # Save the results to a CSV file
    # df.to_csv("Testing_Score.csv", index=False)

    try:
        with conn.cursor() as cursor:
            # Prepare an UPDATE statement
            update_query = """
            UPDATE source_file
            SET scorecard = %s
            WHERE "Organization Name" = %s
            """
            
            # Execute updates for each row
            for index, row in df.iterrows():
                cursor.execute(update_query, (row['Scorecard_1'], row['Organization Name']))
            
            # Commit the transaction
            conn.commit()
            print(f"Successfully updated {cursor.rowcount} rows with scorecards.")
    
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        print(f"Error updating database with scorecards: {e}")
    
    return df

    # return df





# pipe = pipeline()

# pipe ={
#     'data_preprocessing':data_preprocessing,
#     'encoded_data_fund_type':encoded_data_fund_type
#     }
# }

"""
fetch data from database -> read data -> clean data -> encode data -> drop_col 
->  split data into features and target -> verify that both df and ml-features have same columns
-> ensure the sequence of columns is the same in both df and ml_features -> Start ml pickle model
"""

if __name__ == "__main__":
    conn = connect()
    if conn is not None:
        table_name='source_file'
        # df1 = run_query(tablename=table_name,conn=conn)
        # print(df1.head())
        data = query(conn,table_name=table_name)
    # describe_data(data)
    print("Step1 Done")
    drop_col(data)
    print('Dropping the col done.')
    # remove_Duplicates(data)
    print("Removing the duplicates done.")
    year_creation(data)
    print("Year creation done ")
    transform_df(data)
    print("Transform done")
    
    feature_engineering(data)
    print("Feature engineering done")
    # transform_df(data)
    scale_data(data)
    print("Scaling done")
    data,encoded_emp_list = encode_emp(data)
    encode_emp_to_numeric(data,encoded_emp_list)
    print("PASSED")
    encode_fund_type(data)
    print("Fund type encoded")
    last_preprocessing(data)
    
    print(data.head(5))
    # last_col_drop(data)
    # split_features_target(data)
    model = get_model("latest_xgboost_model.pkl")
    predict_score(data,model,ml_features)
    
