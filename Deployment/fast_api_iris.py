import pickle
import sklearn
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

MODEL_FILE = "model.pkl"

my_model = pickle.load(open(MODEL_FILE, 'rb'))

#load the dataset from prediction_dataset.csv
class_names = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]

@app.get("/iris_model/")
def knn_classifier(sepal_length, sepal_width, petal_length, petal_width):
    """ will to stuff to your request """
    if sepal_length and sepal_width and petal_length and petal_width is not None:

        data = f'{sepal_length},{sepal_width},{petal_length},{petal_width}'
        input_data = list(map(float, data.split(',')))
        prediction = my_model.predict([input_data]).tolist()
        ## return the prediction and the name of the class
        ## like this:{"prediction": "Iris-Setosa","prediction_int": 0}
        name = class_names[prediction[0]]
        return {"prediction": name, "prediction_int": prediction[0]}
    else:
        return 'No input data received'


from fastapi import BackgroundTasks
from datetime import datetime

with open('prediction_database.csv', 'w') as file:
    file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")

def add_to_database(
    now: str, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float, prediction: int):
    with open('prediction_database.csv', 'a') as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n")

@app.get("/iris_model_v2/")
def knn_classifier(sepal_length, sepal_width, petal_length, petal_width, background_tasks: BackgroundTasks):
    """ will to stuff to your request """
    if sepal_length and sepal_width and petal_length and petal_width is not None:

        data = f'{sepal_length},{sepal_width},{petal_length},{petal_width}'
        input_data = list(map(float, data.split(',')))
        prediction = my_model.predict([input_data]).tolist()
        ## return the prediction and the name of the class
        ## like this:{"prediction": "Iris-Setosa","prediction_int": 0}
        name = class_names[prediction[0]]

        now = str(datetime.now())
        background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width, prediction[0])

        return {"prediction": name, "prediction_int": prediction[0]}
    else:
        return 'No input data received'


