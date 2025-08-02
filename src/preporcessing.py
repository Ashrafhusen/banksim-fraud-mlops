import pandas as pd  
from sklearn.preprocessing import LabelEncoder
import os  
import numpy as np 

def processing_data(input_path : str, output_path: str):
    df = pd.read_csv(input_path)
    
    if 'zipcodeOri' in df.columns:
        df.drop(['zipcodeOri', 'zipMerchant'], axis = 1, inplace = True)

    cat_cols = ['customer', 'merchant', 'category', 'gender']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    df['amount_log']  = df['amount'].apply(lambda x: np.log1p(x))
    df['hours_of_day'] = df['step'] % 24;

    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df.to_parquet(output_path, index = False)
    print(f"Data Processed to {output_path}")

    return df 

if __name__ == "__main__":
    processing_data("data/raw.csv", "data/processed/processed.parquet")
