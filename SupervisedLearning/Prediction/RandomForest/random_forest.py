# random forest
# rassal ağaçlar/orman algoritması

# ensemble learning : kollektif öğrenme
# birden fazla sınıflandırma veya tahmin uygulaması aynı anda kullanılarak daha başarılı sonuç çıkartır
# random forest birden fazla decision treenin aynı problem için kullanılmasıdır
# veri kümesini küçük parçalara böler ve hepsi için ayrı decision tree oluşturur
# sonrasında sınıflandırmada majority voted learning kullanılarak çoğunluğun değerini alır
# tahminde ise sonuçların ortalamasını alır
# decision tree veriler üzerinde quantile kutulama yapar


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('data.csv')

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

Z = X + 0.5
K = X - 0.4

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.show()
