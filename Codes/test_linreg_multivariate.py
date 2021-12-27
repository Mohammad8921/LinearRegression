'''
    TEST SCRIPT FOR MULTIVARIATE LINEAR REGRESSION
'''

'''
Numpy is a standard library in python that lets you do matrix and vector operations like Matlab in python.
Check out documentation here: http://wiki.scipy.org/Tentative_NumPy_Tutorial
If you are a Matlab user this page is super useful: http://wiki.scipy.org/NumPy_for_Matlab_Users 
'''
import numpy as np
from numpy.linalg import *
from math import sqrt

# our linear regression class
from linreg import LinearRegression


if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    
    # load the data
    filePath = "data/multivariateData.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
    lr_model.fit(X,y)
    print("=================== Predict The Test set ======================")
    dataset = np.load("./data/holdout.npz")
    dataset = dataset['arr_0']
    X_test = np.matrix(dataset[:, :-1])
    y_test = np.matrix(dataset[:, -1]).T
    n_test = X_test.shape[0]
    # Standardize Independent Variables in Test set
    mean_test = X_test.mean(axis=0)
    std_test = X_test.std(axis=0)
    X_test = (X_test - mean)/std
    X_test = np.concatenate((np.ones((n_test,1)), X_test), axis=1)
    print("Model Predited Values for Dependent Variable of Test Set:", lr_model.predict(X_test), sep='\n')
    print("Error of the Test set: ", sqrt(lr_model.computeCost(X_test, y_test, lr_model.theta)*2), sep='\n')
    


