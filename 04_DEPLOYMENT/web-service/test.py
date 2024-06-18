# import predict
import requests
ride = {
    "PULocationID" : 10,
    "DOLocationID" : 50,
    "trip_distance" : 40
}
url = 'http://127.0.0.1:9595/predict'
response = requests.post(url, json=ride)
# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred)
print(response.json())