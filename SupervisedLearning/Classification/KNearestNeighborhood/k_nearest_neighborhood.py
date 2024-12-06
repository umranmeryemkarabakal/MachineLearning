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

#Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)


# knn k nearest neighborhood
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


#confusion matrix for knn
cm = confusion_matrix(y_test,y_pred)
print(cm)


