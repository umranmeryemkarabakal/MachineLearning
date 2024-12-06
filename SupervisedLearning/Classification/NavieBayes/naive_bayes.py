# naive Bayes : koşullu olasılık dayalı bir sınıflandırma yöntemidir. Koşullu olasılık şu şekilde hesaplanır:

# P(A|B) = P(AkesişimB) / P(B)

# öğrenme Yöntemleri
# lazy learning : tembel Öğrenme : Yeni veri geldiğinde öğrenme işlemi yapılır, veriler üzerinde işlem anında gerçekleştirilir. Örnek: K-en Yakın Komşu (KNN).
# eager learning : istekli,Hevesli Öğrenme : Yeni veri gelmeden önce, eldeki veri kümesiyle bir model oluşturulur ve öğrenme işlemi gerçekleşir. Örnek: Naive Bayes, Karar Ağaçları.

# gaussian naive bayes : sürekli değişkenlerin tahmini için kullanılır. Değişkenler normal dağılıma sahip olarak kabul edilir.
# multinomial naive bayes : nominal verilerin veya tam sayılarla ifade edilen kategorik değişkenlerin tahmini için kullanılır.
# bernoulli naive bayes : ikili sonuçlar (örneğin 0-1) için tahmin yapmak amacıyla kullanılır.


#gnb
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


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

