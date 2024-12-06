# recognition (%) = CC / total
# Tanıma oranı (%) = Doğru sınıflandırma (CC) / Toplam

# error rate - misclassification rate 
# Hata oranı - yanlış sınıflandırma oranı

# e.g. for cancer diagnosis: alternative metrics 
# Örneğin, kanser teşhisi için: alternatif ölçütler
# sensitivity, specificity, precision, accuracy 
# duyarlılık, özgüllük, kesinlik, doğruluk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('data.csv')

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#lojistic regression
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

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


# ROC , TPR, FPR değerleri 
y_proba = rfc.predict_proba(X_test)
# X_test üzerindeki tahmin olasılıklarını hesaplar. predict_proba fonksiyonu, sınıflar için olasılık tahminleri döndürür. Sonuç iki boyutlu bir numpy dizisidir.
print("-------")
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e') #pos label pozitif olarak alacak olan değer
print("-------")
print(fpr)
print(tpr)

# True Positive (TP): Modelin pozitif olarak tahmin ettiği ve gerçekte de pozitif olan örnekler.
# True Negative (TN): Modelin negatif olarak tahmin ettiği ve gerçekte de negatif olan örnekler.
# False Positive (FP) (Type I Error): Modelin pozitif tahmin ettiği ancak gerçekte negatif olan örnekler (yanlış pozitif).
# False Negative (FN) (Type II Error): Modelin negatif tahmin ettiği ancak gerçekte pozitif olan örnekler (yanlış negatif).

# roc uzayı , algoritmanın eşik değeri hiperparametresi varsa optimum çalışma noktasının bulunmasını sağlar
# farklı sınıflandırıcıların performanslarının birbiriyle karşılaştırılmasını sağlar
 
# Zero-R algoritması: En yaygın sınıfı tahmin eder
# Bir algoritmanın Zero-R algoritmasından daha iyi sonuç vermesi beklenir
