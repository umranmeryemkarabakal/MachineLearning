# svm support vector machine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')

x = data.iloc[:,1:4].values #bağımsız değişkenler
y = data.iloc[:,4:].values #bağımlı değişken

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#svc
from sklearn.svm import SVC
svc = SVC(kernel='poly') # çekirdek trick'i polinom çekirdek ile kullanılır
# Kernel trick: Veriler doğrusal değilse, polinom çekirdeği ile veriler daha yüksek boyutlara taşınarak doğrusal hale getirilir.

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)