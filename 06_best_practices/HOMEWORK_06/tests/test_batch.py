from  hw_03_batch import prepare_data
from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_data_prep():
    data = [
            (None, None, dt(1, 1), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, None, dt(1, 2, 0), dt(1, 2, 59)),
            (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
            ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns) 

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical=categorical)

    expected_raw_data= [
        {'PULocationID' : '-1', 
            'DOLocationID' : '-1',
                'duration' : 9.0 },
        {'PULocationID' : '1', 
            'DOLocationID' : '1',
                'duration' : 8.0}]
    
    expected_df = pd.DataFrame(expected_raw_data)
    actual_df_comp = actual_df[categorical + ['duration']] 
    pd.testing.assert_frame_equal(actual_df_comp, expected_df)





