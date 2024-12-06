import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data1.csv",sep=";")
df.head(3)

X = df.iloc[:,[0,2]].values
y = df.salary.values.reshape(-1,1)

model = LinearRegression()

model.fit(X,y)

experience = 15
age =35
person =np.array([[experience,age]])
new_salary = model.predict(person)
print("The salary of a person with {} years of experience and {} years of age is {} TL ".format(experience,age,new_salary[0]))

from sklearn.metrics import r2_score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2}")

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")


# mean absolute error (MAE): Tahmin edilen değerler ile gerçek değerler arasındaki mutlak farkların ortalaması.
# mean squared error (MSE): Tahmin edilen değerler ile gerçek değerler arasındaki farkların karelerinin ortalaması.
# root mean squared error (RMSE): MSE'nin kareköküdür ve hataları daha anlaşılır bir ölçüye dönüştürür.
# R-squared (R²): Modelin verileri ne kadar iyi açıkladığını gösteren bir metriktir. 1'e yakın olması modelin iyi bir performans sergilediğini gösterir.
