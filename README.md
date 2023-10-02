# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: J.NETHRAA
RegisterNumber:  212222100031
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)


```

## Output:

### Array Value of x
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/abe2f5cc-18c4-484b-975f-c8403b290291)

### Array Value of y
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/530f0deb-2097-4570-b668-cb1b9f854055)

### Exam 1 - score graph
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/4db5ab56-0106-42f2-b099-8d6f4c6dbfc9)

### Sigmoid function graph
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/8b16410f-b45d-4e54-a921-886497809cfb)

### X_train_grad value
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/1c017ddd-9309-44a3-9aa7-21fa85eae834)

### Y_train_grad value
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/f528308c-5870-48c3-97a2-6bd67d7df1ca)

### Print res.x
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/f95a558c-77a7-4c52-abda-705e9b37b4e2)

### Decision boundary - graph for exam score
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/ade9aab1-a863-4794-9cb3-c6643aa95609)

### Probablity value
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/08118001-4969-4c9d-b44e-1a9739491f83)

### Prediction value of mean
![image](https://github.com/Nethraa24/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215786/b1f3202a-5a00-49cb-bb46-b306adb7bc05)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

