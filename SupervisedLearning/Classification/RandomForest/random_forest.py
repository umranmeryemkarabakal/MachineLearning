# ensemble learning (kolektif öğrenme) ile sınıflandırma yapılır
# birden fazla modelin çıktısı birleştirilerek daha güçlü tahminler yapılır

# Real-time human pose recognition: İnsan vücut pozunun tekil derinlik görüntülerinden tanınması
# bu yaklaşım insanın farklı uzuvlarını tespit eder

# random forest
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

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

