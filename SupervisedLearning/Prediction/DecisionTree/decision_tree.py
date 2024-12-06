# decision tree
# karar ağacı ile tahmin

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('data.csv')

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values


r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
#bütün değerleri aynı sonuç olarak predict yaptı
plt.show()
