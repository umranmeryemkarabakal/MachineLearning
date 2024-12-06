# ARM / ARL : association rule mining,learning : birliktelik kural çıkarımı : ilişkilendirme kuralı madenciliği

# tekrar eden eylemi(transaction) yakalamak için kullanılır
# recommender tavsiye algoritmaları 

# correlation is not causation 
# ilişkisellik mi nedensellik mi

# destek - support(a) = a varlığını içeren eylemler / toplam eylem sayısı
# güven - confidence(a->b) = a ve b varlığını içeren eylemler / a varlığını içeren eylemler
# kaldıraç,asansör - lift(a->b) = confidence(a->b) / support(b)

# complex event processing
# kampanya, davranış tahmini
# yönlendirilmiş arm
# zaman serisi analizi


# apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('basket.csv',header = None)

t = [] # transaction listesi
for i in range (0,7501):
    t.append([str(data.values[i,j]) for j in range (0,20)])

from apyori import apriori
rules = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(rules))
#relation record oluşturur


# map reduce 
# büyük veri setlerinde apriori algoritmasının paralelleştirilmesi için MapReduce kullanılır