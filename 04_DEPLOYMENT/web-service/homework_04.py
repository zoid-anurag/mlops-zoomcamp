#!/usr/bin/env python
# coding: utf-8
#get_ipython().system('pip freeze | grep scikit-learn')
#get_ipython().system('python -V')

import pickle 
import pandas as pd
import sys

with open('./model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

if len(sys.argv) < 3:
    print("Use : <filename.py> <year> <month>", file=sys.stderr)
    exit(1)

year = int(sys.argv[1])
month = int(sys.argv[2])

print(year, month)

# Validate year and month
if year < 2009 or year > 2024:
    print("Wrong Year", file=sys.stderr)

if month > 12 or month < 1:
    print("invalid month", file=sys.stderr)


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet
print(df.shape)
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
print("Mean predicted duration is : {}".format(y_pred.mean()))
# print(np.std(y_pred))
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# # df['ride_id'].head()
# df_result = pd.DataFrame()
# df_result['ride_id'] = df['ride_id']
# df_result['predictions'] = y_pred


# print(df_result.shape)
# print(df_result.tail())
# output_file = "prediction_results"
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )






