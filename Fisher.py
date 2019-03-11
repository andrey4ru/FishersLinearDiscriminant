import Math
import numpy as np

class Fisher:

    def Train(self, data, label):
        M = Math.Expectation(data, label)
        M_0 = np.array(M[0])
        M_1 = np.array(M[1])
        self.__S_w = Math.Scatter_Matrix(data, label, M_0, M_1)  # inverted
        self.__W = np.dot(self.__S_w, M_0 - M_1)
        Y_0 = np.dot(self.__W, M_0)
        Y_1 = np.dot(self.__W, M_1)
        self.__delta = (Y_0 + Y_1)/2


    def Predict(self,data):
        predict = []
        for i in range(len(data)):
            X = np.array(data.iloc[i])
            predict.append(np.dot(self.__W, X)-self.__delta)

        return predict


