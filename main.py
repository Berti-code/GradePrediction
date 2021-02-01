import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

prediction_label = "G3"

# get all the data except the prediction_label column and the 1st index column
features = np.array(data.drop([prediction_label], 1))
# get all the values for the prediction_label
g3_values = np.array(data[prediction_label])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, g3_values, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print("Model accuracy= ", accuracy)

predictions = linear.predict(x_test)  # get all the predictions

for x in range(len(predictions)):
    if predictions[x] < 0 : predictions[x] = 0
    else : predictions[x] = int(predictions[x])
    print("Feature values: ", x_test[x], "\t", "Actual grade for G3= ",  y_test[x], "\t", "Prediction for G3= ", predictions[x])