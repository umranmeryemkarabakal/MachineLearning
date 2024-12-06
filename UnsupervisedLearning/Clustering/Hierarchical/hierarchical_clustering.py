# agglomerative : bottom up approach :aşağıdan yukarı çalışan sistemdir
# disive : top down approach : yukarıdan aşağı giden sistemdir

# agglomerative
# Hhr veri noktası başlangıçta bir küme olarak kabul eder
# en yakın iki komşu bulunur ve yeni bir küme oluşturur
# tek bir küme kalana kadar devam eder

# mesafe ölçümü 
# 1.metrikler
# -öklit mesafesi
# 2.referanslar
# -en yakın noktalar -mim
# -en uzak noktalar -max
# -ortalama, ağırlık merkezi -
# -merkezler arası mesafe -group average

# ward's mwthod : mesafe metriğidir
# Distance matrix : mesafe matrisi
# nested cluster : iç içe kümeleme
# dendogram
# en optimum nokta mesafenin en  yüksek olduğu noktasır


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data.iloc[:,3:].values


#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)  
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()


#HC hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)


plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# dendogram tanımlanıp ward mesafesi kullanarak linkage-bağlantı yapar
plt.show()
