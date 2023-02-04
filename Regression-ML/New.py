import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# 1-

data = pd.read_csv('trainRegression.csv') 
data.head()

testData = pd.read_csv('testRegression.csv') 
testData.head()

plt.plot(data['X'],data['R'],'*')




# 2-
    # For Linear Model theta0, theta1******************************

A = np.array( [ [ data.shape[0], sum(data['X']) ],
               [ sum(data['X']), sum(data['X']**2)] ])

B = np.array( [ [sum(data['R']) ],
               [ sum(data['X'].mul(data['R']) ) ]])

X = np.dot(np.linalg.inv(A) , B)
ltheta0 = X[0,0]
ltheta1 = X[1,0]

pRl = ltheta0 + (ltheta1*data['X'])
plt.plot(data['X'],pRl,'*')

mseL = sum((data['R'] - pRl)**2)/data.shape[0]
print("Mean Square Error for Linear Model = ", mseL )
# print("theta Matrix for Linear Model : \n", X)
print("\n***\n")

 # Using Test data for linear Model
pRlTest = ltheta0 + (ltheta1*testData['X'])




     # For Quadratic Model theta0, theta1, theta2*************************

A1 = np.array([[data.shape[0], sum(data['X']), sum(data['X']**2)],
             [sum(data['X']), sum(data['X']**2), sum(data['X']**3)],
             [sum(data['X']**2), sum(data['X']**3), sum(data['X']**4)]])

B1 = np.array([[sum(data['R'])],
              [sum(data['R'].multiply(data['X']))],
              [sum(data['R'].multiply(data['X']**2))]])

X1 = np.dot(np.linalg.inv(A1) , B1)
qtheta0 = X1[0,0]
qtheta1 = X1[1,0]
qtheta2 = X1[2,0]
# print("\nQTHETAAS: ", qtheta0,"\t", qtheta1, "\t", qtheta2)
# pRq = qtheta0 + (qtheta1*data['X']) + (qtheta2 * data['X']**2)
pRq = qtheta0 + (qtheta1 * data['X']) + (qtheta2 * (data['X']**2) )
plt.plot(data['X'],pRq,'*')
mseQ = sum((data['R'] - pRq)**2)/data.shape[0]
print("Mean Square Error for Quadratic Model = ", mseQ )
# print("theta Matrix for Quadratic Model : \n", X1)
print("\n***\n")

# print(A1)
# print("\n***\n")
# print(B1)
# print("\n***\n")

 # Using Test data for Quadratic Model
pRqTest = qtheta0 + (qtheta1*testData['X']) + (qtheta2*testData['X']**2)
    




     # For Cubic Model theta0, theta1, theta2, theta3 **********************
     
A2 = np.array([[len(data), sum(data['X']), sum(data['X']**2), sum(data['X']**3)],
             [sum(data['X']), sum(data['X']**2), sum(data['X']**3), sum(data['X']**4)],
             [sum(data['X']**2), sum(data['X']**3), sum(data['X']**4), sum(data['X']**5)],
             [sum(data['X']**3), sum(data['X']**4), sum(data['X']**5), sum(data['X']**6)]])

B2 = np.array([[sum(data['R'])],
              [sum(data['R'].mul(data['X']))],
              [sum(data['R'].mul(data['X']**2))],
              [sum(data['R'].mul(data['X']**3))]])

X2 = np.dot(np.linalg.inv(A2) , B2)
ctheta0 = X2[0,0]
ctheta1 = X2[1,0]
ctheta2 = X2[2,0]
ctheta3 = X2[3,0]

pRc = ctheta0 + (ctheta1*data['X']) + (ctheta2*data['X']**2) + (ctheta3*data['X']**3)
plt.plot(data['X'],pRc,'*')
mseC = sum((data['R'] - pRc)**2)/data.shape[0]
print("Mean Square Error for Cubic Model = ", mseC )

    # USing Test Data for Cubic Model 
pRcTest = ctheta0 + (ctheta1*testData['X']) + (ctheta2*testData['X']**2) + (ctheta3*testData['X']**3)

# print("theta Matrix for Cubic Model : \n", X2)
print("\n***\n")




    # 6- Plotting The Testing data and predictions
figure, axis = plt.subplots(1, 1)
axis.plot(testData['X'],testData['R'],'*')
axis.plot(testData['X'],pRlTest,'*')
axis.plot(testData['X'],pRqTest,'*')
axis.plot(testData['X'],pRcTest,'*')

axis.set_title("For Test Data")



    # 7- Mean Sqaure Errors for Testing Data
#Linear Model
mseLTest = sum((testData['R'] - pRlTest)**2)/testData['R'].size
print("Mean Square Error for Linear Model (Testing Data) = ", mseLTest )



#Quadratic Model
mseQTest = sum((testData['R'] - pRqTest)**2)/testData['R'].size
print("Mean Square Error for Quadratic Model (Testing Data) = ", mseQTest )

#Cubic Model
mseCTest = sum((testData['R'] - pRcTest)**2)/testData['R'].size
print("Mean Square Error for Cubic Model (Testing Data) = ", mseCTest )

# print("A1 DETERMINANACT", np.linalg.det(A1),)
# print("A DETERMINANACT", np.linalg.det(A))
# print("A2 DETERMINANACT", np.linalg.det(A2))
print("\n", A1, "\n")
print("\n", A2, "\n")
print(A)
print(type(data.shape[0]), data.shape[0])
print(type(data['R'].size), data['R'].size )
#   8. Comment on the results

#   For the Given Data, Cubic model is performing best, as its Mean Square Error is lowest and also visually we can see 
#   in the plotted graphs, that cubic model is plotting best. For Training data Quadratic model is slightly better than
#   linear model. But for testing data Quadratic Model is performing worst.