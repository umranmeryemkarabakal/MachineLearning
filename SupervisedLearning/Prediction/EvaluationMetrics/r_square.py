#evaluation of prediction

# r-square : r-kare yöntemi : modelin verileri ne kadar iyi açıkladığını gösterir
# 1. hata kareleri toplamı = sum(yi - y'i)^2  y-tahmin
# 2. ortalama farkların toplamı = sum(yi - yort)^2 base line: taban çizgisi / zero r belirtir olabilecek en kötü değer
# r_2 = 1- (hkt / oft)

# yeni eklenen değişkenler sisteme olumsuz etkisi olsa dahi r_2 değerini düşürmez , olumlu etkisi varsa r_2 değerini arttırır
# adjusted r_2 : düzeltilmiş r_2 hesaplaması 
# y = a0x0 + a1x1 + a2x2 + b
# n : kaç eleman olduğu
# p : kaç değişken alındığı
# adjusted r^2 = 1 - (1 -r^2) * ((n-1)/ (n-p-1) )

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

data = pd.read_csv('data.csv')

x = data[['Egitim Seviyesi']]  
y = data['maas']
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.title("linear regression")
plt.show()

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("polynomial regression degree:2")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("polynomial regression degree:4")
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.title("support vector regression")
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.title("decision tree regression")
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.title("random forest regression")
plt.show()

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))

#Ozet R2 değerleri
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

