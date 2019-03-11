def TPR(predict, target, threshold, class_numb):
    TP = 0  # true positive
    CP = 0  # condition positive
    for i in range(len(predict)):
        # print(target[i], class_numb, target[i] == class_numb)
        if target[i] == class_numb:
            CP += 1  # calculate number of condition positive
            if predict[i] >= threshold:
                TP += 1  # calculate number of true positive
    return TP/CP  # return true positive rate

