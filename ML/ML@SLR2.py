import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'D:\Data science\FSDS 4pm\Class Notes\8th July\7th- slr\SLR - House price prediction\House_data.csv')
dataset

space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the prices
pred = regressor.predict(x_test)

#Visualizing the training Test Results 
plt.scatter(x_train, y_train, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)
