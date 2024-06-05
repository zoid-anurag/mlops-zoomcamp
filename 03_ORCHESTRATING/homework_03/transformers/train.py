import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    categorical = ['PULocationID','DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = data[categorical].to_dict(orient = 'records')
    X_train = dv.fit_transform(train_dicts)

    # val_dicts = df_val[categorical + numerical].to_dict(orient= 'records')
    # X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = data[target].values
    # y_val = df_val[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # y_pred = lr.predict(X_val)

    # mean_squared_error(y_train, y_pred, squared=False)
    print(lr.intercept_)


    return lr, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'