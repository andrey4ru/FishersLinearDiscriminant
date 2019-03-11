import matplotlib.pyplot as plt
import numpy as np
import TPR
import FPR


def ROC(predict, target, class_numb):
    predict = np.array(predict)  # to numpy array
    predict = (predict - min(predict))/(max(predict) - min(predict))  # normalize to (0, 1)
    n = 10  # plot discretization
    tpr = np.zeros(n)
    fpr = np.zeros(n)
    step = max(predict)/(n - 1)  # step size
    for i in range(n):
        tpr[i] = TPR.TPR(predict, target, i * step, class_numb)
        fpr[i] = FPR.FPR(predict, target, i * step, class_numb)
    plt.plot(fpr, tpr, color='Blue', linewidth=1)  # plotting
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR(Sensitivity)')
    plt.show()
