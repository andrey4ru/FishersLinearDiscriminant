import numpy as np


def Expectation(data, lable):  # calculating expected value
    M_0 = [0] * 4
    M_1 = [0] * 4
    res = []
    n_0, n_1 = 0, 0
    for j in range(len(data.iloc[0]) - 1):
        for i in range(len(data)):
            if data.iloc[i, -1] == lable[0]:
                M_0[j] += data.iloc[i, j]
                n_0 += 1
            else:
                M_1[j] += data.iloc[i, j]
                n_1 += 1
    n_0 /= len(data.iloc[0]) - 1
    n_1 /= len(data.iloc[0]) - 1
    for i in range(len(data.iloc[0]) - 1):
        M_0[i] = M_0[i]/n_0
        M_1[i] = M_1[i]/n_1
    res.extend([M_0, M_1])
    return res


def Scatter_Matrix(data, lable, M_0, M_1):
    X_0 = [0] * 4   # X_0 = X_0 - M_0
    X_1 = [0] * 4   # X_1 = X_1 - M_1
    n_0, n_1 = 0, 0
    for j in range(len(data.iloc[0]) - 1):
        for i in range(len(data)):
            if data.iloc[i, -1] == lable[0]:
                X_0[j] += data.iloc[i, j] - M_0[j]
                n_0 += 1
            else:
                X_1[j] += data.iloc[i, j] - M_1[j]
                n_1 += 1

    n_0 /= (len(data.iloc[0]) - 1)
    n_1 /= (len(data.iloc[0]) - 1)
    X_0 = np.matrix(X_0)
    X_0_T = X_0
    X_0 = np.transpose(X_0)
    S_0 = np.dot(X_0, X_0_T)/(n_0 - 1)

    X_1 = np.matrix(X_1)
    X_1_T = X_1
    X_1 = np.transpose(X_1)
    S_1 = np.dot(X_1, X_1_T)/(n_1 - 1)

    S_w = ((n_0 - 1) * S_0 + (n_1 - 2) * S_1)/(n_0 + n_1 - 2)
    S_w = np.linalg.inv(S_w)

    return S_w