# support vector regression
# destek vektör regresyonu

# amacı maximum marjin değerini minimalize eden doğruyu bulma
# marjin dışı noktaları hata olarak almaktır

# kernel çeşitleri
# doğrusal : linear
# polinom : polynomial
# guassian : RBF
# üssel : exponential


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVR

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

#svr module
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')


print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
