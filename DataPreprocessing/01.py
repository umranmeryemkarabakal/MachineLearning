import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer #IterativeImputer, KNNImputer 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# data import
data = pd.read_csv("data.csv")

# worker directory
# absulut path --> "C:\\Users\\Desktop\\MachineLearning\\data.csv"
# relavie path --> "data.csv"

height = data[["height"]] # list of lists
#print(height)

height_weight = data[["height","weight"]]
#print(height_weight)


# missing values
# imputation
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# boş değerleri ortalama ile doldurur

age = data.iloc[:,1:4].values
#print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
#print(age)

# fit ile öğrenilir , transform ile öğrenileni uygular


# type transform
country = data.iloc[:,0:1].values
#print(country)

label_encoding = preprocessing.LabelEncoder()
# her bir kategorik veriye tam sayı atayarak sayısal veriye dönüştürür

country[:,0] = label_encoding.fit_transform(data.iloc[:,0])
#print(country)

one_hot_encoder = preprocessing.OneHotEncoder()
# her kategorik veri için sütunlara mevcutluk durumuna göre 1-0 olarak ayırır

country = one_hot_encoder.fit_transform(country).toarray()
#print(country)


# merge data frame 
result = pd.DataFrame(data=country, index=range(22), columns=["fr","tr","us"])
# data frame diziden farklı olarak index kolonu ve kolon başlıkları bulunur
#print(result)

result2 = pd.DataFrame(data=age, index=range(22), columns= ["height","weight","age"])
#print(result2)

gender = data.iloc[:,-1].values
result3 = pd.DataFrame(data=gender, index=range(22), columns=["gender"])
#print(result3)

s = pd.concat([result, result2],axis=1) 
# sütun isimlerine göre eşleme yapar

s2 = pd.concat([s, result3],axis=1)
#print(s2)


# data split for train and test  
x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)


# attribute standardization
sc = StandardScaler()
# verilerin ortalamasını 0 ve standart sapmasını 1 olacak şekilde ölçekler.
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
print(X_test)
print(X_train)
