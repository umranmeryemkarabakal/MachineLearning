# prediction : tahmin
# forecastion : öngörü,zaman serisinde daha sonrasının tahmin edilmesi

# simple linear regression
# y = ax + b + e
# y: dependent variables, x: independent variables , a: coefficient, e: error term

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv')
#print(data)

months = data[["months"]]
sales = data.iloc[:,:1].values

x_train, x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

# linear regression
lr = LinearRegression()
lr.fit(x_train,y_train)

prediction_lr = lr.predict(x_test)

#plt.plot(x_train,y_train) # veriler sıralı değildir
x_train = x_train.sort_index()
y_train = pd.DataFrame(y_train, index=x_train.index).sort_index()

plt.plot(x_train, y_train, label="Training data")
plt.plot(x_test, lr.predict(x_test), label="Test predictions", color='red')

plt.title("sales by month")
plt.xlabel("months")
plt.ylabel("sales")

plt.legend()
plt.show()
