# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#here we dn't need to split the dataset coz we have less number of data..
#we should me more accurate about the salary

#no need to apply feature scaling coz library will do that work like in multiple linear Regression

#fitting the linear regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting the polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

#fitting the  multipl linear regression model to the dataset which is transformed into polynomial
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#visualizing the linear regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.title("Truth Or Bluff{LinearRegressiom}")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

#vaisualizing the polynomial regression which is better than linear Regression
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title("Truth Or Bluff{polynomial Regressiom}")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

#predicting the new result with linear regression
lin_reg.predict(6.5)

#predicting the new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))