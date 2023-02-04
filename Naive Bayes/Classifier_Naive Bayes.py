import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt

img2=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.uint8)

img4=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.uint8)

img = np.reshape(img2,(16,16),order='F')

plt.imshow(img)

def class_probabilities(one_given2, zero_given2, data_main, imagesCount):
    probabilities = []
    for i in range(imagesCount):
        prob_2 = 0.5
        for x in range(256):
            if data_main[i][x] == 1:
                prob_2 *= one_given2[x]
            elif data_main[i][x] == 0:
                prob_2 *= zero_given2[x]
        probabilities.append(prob_2)
    return probabilities

def classifier(prob2, prob4, imagesCount):
    pred_classes = []
    for i in range(imagesCount):
        if prob2[i] > prob4[i]:
            pred_classes.append(2)
        else:
            pred_classes.append(4)
    return pred_classes
        
def overall_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual)

def class_accuracy(actual, predicted, imageCount):
    correct_2 = 0
    correct_4 = 0
    
    for i in range( int(len(actual)/2) ):
        if actual[i] == predicted[i]:
            correct_2 += 1
    for i in range( int(len(actual)/2), imageCount ):
        if actual[i] == predicted[i]:
            correct_4 += 1
            
    two_class_acc = correct_2 / (int(len(actual)/2))
    four_class_acc = correct_4 / (int(len(actual)/2))
    
    return two_class_acc, four_class_acc


trainX = np.loadtxt('trainX.txt')
trainY = np.loadtxt('trainY.txt')

testX = np.loadtxt('testX.txt')
testY = np.loadtxt('testY.txt')


half = int(trainX.shape[0] / 2)
data2 = trainX[:half, :]
# print(data2.shape)

data4 = trainX[half:, :]
# print(data4.shape)

onegiven2 = (data2.sum(axis = 0) + 1) / ( data2.shape[0] + 2)
zerogiven2 = 1 - onegiven2


onegiven4 = (data4.sum(axis = 0) + 1 ) / ( data4.shape[0] + 2)
zerogiven4 = 1 - onegiven4

class2_train_pred = class_probabilities(onegiven2, zerogiven2, trainX, 500)
class4_train_pred = class_probabilities(onegiven4, zerogiven4, trainX, 500)
pred_classes = classifier(class2_train_pred, class4_train_pred, 500)
accuracy_o = overall_accuracy(trainY, pred_classes)
print("Overall Accuracy for Train Data :", accuracy_o)

class2_test_pred = class_probabilities(onegiven2, zerogiven2, testX, 100)
class4_test_pred = class_probabilities(onegiven4, zerogiven4, testX, 100)
pred_classes_t = classifier(class2_test_pred, class4_test_pred, 100)
accuracy_t = overall_accuracy(testY, pred_classes_t)
print("Overall Accuracy for Test Data :", accuracy_t)

class2_acc, class4_acc = class_accuracy(trainY, pred_classes, 500)
print("Class 2 Accuracy for Training Data:",class2_acc, "\t Class 4 Accuracy for Training Data", class4_acc )

class2_acc_t, class4_acc_t = class_accuracy(testY, pred_classes_t, 100)
print("Class 2 Accuracy for Test Data:", class2_acc_t,"\t Class 4 Accuracy for Test Data", class4_acc_t)

