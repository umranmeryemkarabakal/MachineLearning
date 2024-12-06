# dummy variable: kukla değişken
# Bir değişkenin başka bir değişken cinsinden temsil edilmesi, genellikle kategorik verilerin sayısal hale getirilmesi

# H0: null hypothesis: farksızlık hipotezi, sıfır hipotezi, boş hipotez
# H1: alternatif hipotez
# p value: olasılık değeri
# p değeri genelde 0.05 alınır
# p değeri küçüldükçe H0'ın reddedilme olasılığı artar, H1'in doğru olma ihtimali yükselir

# Variable selection
# bütün değişkenleri dahil etmek
# stepwise davranışlar
#   backward elimination: geriye doğru eleme
#       1. signifiance level(SL): başarı kriteri berirlenir, genelde 0.05 alınır
#       2. bütün değişkenlerle model inşa edilir
#       3. en yüksek p-value değerine sahip olan değişken ele alınır, şayet p>SL ise 4.adıma, değilse 6.adıma gidilir
#       4. 3.adımda seçilen ve en yüksek p değerine sahip değişken sistemden kaldırılır
#       5. makine öğrenmesi güncellenip 3.adıma geri dönülür
#       6. makine öğrenmesi sonladırılır
#   forward selection: bidirectional elimination
#       1. signifiance level(SL): başarı kriteri berirlenir, genelde 0.05 alınır
#       2. bütün değişkenlerle model inşa edilir
#       3. en düşük p-value değerine sahip olan değişken ele 
#       4. 3.adımda seçilen değişken sabit tutularak yeni birdeğişken daha seçilir ve sisteme eklenir
#       5. makine öğrenmesi güncellenip 3.adıma geri dönülür, şayet en düşük p değerine sahip değişken için p<SL şartı sağlanıyorsa 3.adıma dönülür, sağlanmıyorsa 6. adıma geçilir
#       6. makine öğrenmesi sonladırılır
#   bidirectional elimination: çift yönlü eleme
#       1. signifiance level(SL): başarı kriteri berirlenir, genelde 0.05 alınır
#       2. bütün değişkenlerle model inşa edilir
#       3. en düşük p-value değerine sahip olan değişken ele 
#       4. 3.adımda seçilen değişken sabit tutularak diğer bütün değişkenler sisteme dahil edilir ve en düşük p değerine sahip olan sistem de kalır
#       5. SL değerinin altında olan değişkenler sistemde kalır ve eski değişkenlerden hiçbiri sistemden çıkarılamaz
#       6. makine öğrenmesi sonlanır
# score comparison: skor karşılaştırması
#   1. başarı kriteri berirlenir
#   2. bütün olası makine öğrenmesi modelleri inşa edilir, ikili seçim olur
#   3. 1.adımda berirlenen kriterleri en iyi sağlayan yöntem seçilir
#   4. makine öğrenmesi sonlandırılır

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# multiple linear regression
# y = b0 + b1x1 + b2x2 + b3x2 + e

data = pd.read_csv('C:\\Users\\umk54\\OneDrive\\Masaüstü\\Machine Learning\\Prediction\\MultiLinearRegression\\data.csv')

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

age = data.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])

label_encoding = preprocessing.LabelEncoder()

country = data.iloc[:,0:1].values
country[:,0] = label_encoding.fit_transform(data.iloc[:,0])

label_encoding = preprocessing.LabelEncoder()

g = data.iloc[:,0:1].values
g[:,0] = label_encoding.fit_transform(data.iloc[:,0])

one_hot_encoder = preprocessing.OneHotEncoder()

country = one_hot_encoder.fit_transform(country).toarray()


result = pd.DataFrame(data=country, index=range(22), columns=["fr","tr","us"])
result2 = pd.DataFrame(data=age, index=range(22), columns= ["height","weight","age"])
gender = data.iloc[:,-1].values
result3 = pd.DataFrame(data=g[:,:1], index=range(22), columns=["gender"])

s = pd.concat([result, result2],axis=1) 
s2 = pd.concat([s, result3],axis=1)

x_train, x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)



height = s2.iloc[:,3:4].values

right = s2.iloc[:,:3]
left = s2.iloc[:,4:]

data = pd.concat([right,left],axis=1)

x_train, x_test,y_train,y_test = train_test_split(data,height,test_size=0.33, random_state=0)

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)


#backward elimination
X = np.append(arr = np.ones((22,1)).astype(int),values=data,axis=1)

X_list = data.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list,dtype=float)

model = sm.OLS(height, X_list).fit()
#print(model.summary())

#OLS: Ordinary Least Squares: tahmin değerleri ile gerçek değerler arasındaki farkların karelerinin toplamını minimize eder.


X_list = data.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list,dtype=float)

model = sm.OLS(height, X_list).fit()
#print(model.summary())


X_list = data.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list,dtype=float)

model = sm.OLS(height, X_list).fit()
print(model.summary())
