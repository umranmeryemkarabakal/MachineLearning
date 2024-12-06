# ID3 algoritmasında dallanma yapılırken, en yüksek bilgi kazancına (information gain) sahip değişken seçilerek ilerlenir.
# bilgi kazancı hesaplamak için yöntemler 
# entropi: Bu yöntemde, bilgi kazancı hesaplanırken olasılıkların logaritmik tabanı alınarak belirsizlik ölçülür.
# gini: Entropiden farklı olarak, olasılıkların logaritması yerine her bir sınıfın olasılığının karesi alınarak hesaplama yapılır. Bu yöntem, sınıfların homojenliğini ölçer.

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


from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy') # kriter entropi

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
