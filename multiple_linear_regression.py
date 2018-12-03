#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

#Encoding the independent categorical variable which is state in our datset
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x=x[:,1:]

#Splitting the data into tarining set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting a multiple regression model for the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set Results
y_pred=regressor.predict(x_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

#looping the above statements 
'''def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols = sm.OLS(y, x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_ols.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_ols.summary()
    return x
 
SL = 0.05     #significance level
x_temp = x[:, [0, 1, 2, 3, 4, 5]]     #according to backward elimination placing all predictors initially
X_opt = backwardElimination(X_temp, SL)'''

#Splitting the optimized data into tarining set and test set
x_opt_train,x_opt_test,y_opt_train,y_opt_test=train_test_split(x_opt,y,test_size=0.2,random_state=0)

#fitting a optimized  multiple regression model for the optimized training set
regressor_opt=LinearRegression()
regressor_opt.fit(x_opt_train,y_opt_train)

#predicting the test set Results
y_opt_pred=regressor_opt.predict(x_opt_test)