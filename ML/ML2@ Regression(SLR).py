import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'D:\Data science\FSDS 4pm\Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# To Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# To Visualize the test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

#prediction:1
y_12 = (m_slope*12) + c_intercept
print(y_12)

#prediction:2
y_20 = (m_slope*20) + c_intercept
print(y_20)

#Mean
dataset.mean()

dataset['Salary'].mean()

#Median
dataset.median()

dataset['Salary'].median()

#Mode
dataset.mode()
dataset['Salary'].mode()

dataset.var()

dataset['Salary'].var()

dataset['Salary'].std()

#Coefficient of variance
#for calculating 'COV' of we have to import a library first
from scipy.stats import variation
variation(dataset.values) # this will give cov of entire dataframe

#Correlation
dataset.corr() # corr of entire dataframe

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew() #this will give skewness of entire dataframe

dataset['Salary'].skew()

#Standard error
dataset.sem() # this will give standard error of entire dataframe

dataset['Salary'].sem() # this will give a standard errot of that particular culumn

#Z-score
#for calculating z-score we have to import a library first
import scipy.stats as stats
dataset.apply(stats.zscore) # this will give the z-score for entire dataset

stats.zscore(dataset['Salary'])  # this will give us z-score for that particular column

#Degree of Freedom

a = dataset.shape[0] # this will give us no.of rows
b = dataset.shape[1] # this will give us no. of columns

degree_of_freedom = a-b
print(degree_of_freedom)

#Sum of Squares of Regreesion(SSR)
y_mean = np.mean(y)

SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#Sum of Squares Error(SSE)
y=y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

SST = SSR + SSE
print(SST)

#R-square
r_square = 1- (SSR/SST)
r_square

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

import pickle
# Save the trained model to disk
filename = 'linear_regression_model.pkl'

# Open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print('Model has been pickled and saved as linear_regression_model.pkl')

import os 
print(os.getcwd())
