# marjinal veriler, outlier : beklenmedik veriler

# content based filtering : Kullanıcının beğendiği ürünlerin özelliklerine bakarak benzer özelliklere sahip ürünler öneren yöntemdir.
# collaborative filtering : Benzer kullanıcıların geçmiş tercihlerine dayanarak önerilerde bulunan yöntemdir.

# k küme sayısını parametre olarak alır
# rastgele olarak k tane merkez noktası seçer
# her veriyi en yakın merkez noktasına göre ilgili kümeye atar
# her küme için yeni merkez noktası hesaplar

# x-means verilen aralıkta k için optimum değer için tek tek dener

# wcss her bir bölüt için bölütteki elemanların merkeze olan mesafelerinin kareleri
# within cluster sums of square 
# wcss-number of cluster grafiğinin dirsek noktası optimum k değeridir 

# k değeri elbow method-dirsek methoduyla seçilir
# k =2,3,4,, öbek merkezleri arasındaki mesafe kaydedilir, grafikteki dirsek noktası k olarak alınır
# elbow point : dirsek noktası


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    #inertia,wcss değerleridir

plt.plot(range(1,11),sonuclar)

plt.show()