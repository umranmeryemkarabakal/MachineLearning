# polynomial regression 
# y = b0 + b1x + b2x^2 + ... + bhx^h
# y = b0 + b1x1 + b2x2 + b11x1^2 + b12x1x2 + e

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")

x = data.iloc[:,1:2]
y = data.iloc[:,2:]

print(x)
print("--")
print(y)

#linear regression
from sklearn.linear_model import LinearRegression

# linear regressin
#doğrusal model oluşturma
lin_reg = LinearRegression()

lin_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color = "red")
plt.plot(x.values,lin_reg.predict(x.values))

from sklearn.preprocessing import PolynomialFeatures

#polynomial regression
#doğrusal olmayan nonlinear model
poly_reg = PolynomialFeatures(degree= 4)
x_poly = poly_reg.fit_transform(x.values)
print(x_poly)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y.values)

#3.yöntem
poly_reg = PolynomialFeatures(degree= 4)
x_poly = poly_reg.fit_transform(x.values)
print(x_poly)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y.values)


plt.scatter(x.values,y.values)
plt.plot(x.values,lin_reg_2.predict(poly_reg.fit_transform(x.values)))
# önce polinom formatına -matrisine- çevir sonra çiz

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg_2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.6]])))

#x values : tip dönüşümü dizi dönüşümü data frameden np.array türüne
