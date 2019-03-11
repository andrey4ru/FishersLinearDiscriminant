import pandas as pd
import numpy as np
import Fisher
import ROC


# ---------------------------------------- Data preparation ----------------------------------------
data = pd.read_csv('iris.csv')  # read data from file
# replace setosa, virginica, versicolor with '1', '2', '3'
data = data.replace(['setosa', 'virginica', 'versicolor'], ['1', '2', '3'])

# ---------------------------------------- Setosa vs Versicolor ------------------------------------
dataTrain = data[10:50]  # split data on train set and test set
dataTrain = dataTrain.append(data[50:90], ignore_index=True)
dataTest = data[0:10]
dataTest = dataTest.append(data[90:100], ignore_index=True)

label = dataTest.Species
dataTest.drop(['Species'], axis=1, inplace=True)
model = Fisher.Fisher()  # create object
model.Train(dataTrain, ['1', '3'])  # train model
predict = model.Predict(dataTest)  # predict
ROC.ROC(predict, label, '3')  # build ROC curve

#  ---------------------------------------- Versicolor vs Virginica ---------------------------------
dataTrain = data[100:140]  # split data on train set and test set
dataTrain = dataTrain.append(data[50:90], ignore_index=True)
dataTest = data[90:100]
dataTest = dataTest.append(data[140:150], ignore_index=True)

label = dataTest.Species
dataTest.drop(['Species'], axis=1, inplace=True)
model = Fisher.Fisher()  # create object
model.Train(dataTrain, ['3', '2'])  # train model
predict = model.Predict(dataTest)  # predict
ROC.ROC(predict, label, '2')  # build ROC curve

# ---------------------------------------- Virginica vs Setosa ----------------------------------------
dataTrain = data[100:140]  # split data on train set and test set
dataTrain = dataTrain.append(data[10:50], ignore_index=True)
dataTest = data[140:150]
dataTest = dataTest.append(data[0:10], ignore_index=True)

label = dataTest.Species
dataTest.drop(['Species'], axis=1, inplace=True)
model = Fisher.Fisher()  # create object
model.Train(dataTrain, ['1', '2'])  # train model
predict = model.Predict(dataTest)  # predict
ROC.ROC(predict, label, '1')  # build ROC curve

