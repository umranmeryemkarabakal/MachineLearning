import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veriler = pd.read_csv('weather_data.csv')

# kategorik verileri sayısal verilere dönüştürme
veriler2 = veriler.apply(LabelEncoder().fit_transform)

# kategorik özelliklerin one-hot encoding işlemi
c = veriler2.iloc[:, :1]
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
c = ohe.fit_transform(c)

# One-hot encoded verileri DataFrame'e dönüştürme
havadurumu = pd.DataFrame(data=c, index=range(len(veriler)), columns=['o', 'r', 's'])

# verilerin birleştirilmesi
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1)
sonveriler = pd.concat([veriler2.iloc[:, -2:], sonveriler], axis=1)

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1:], test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print("Tahmin Edilen Değerler:")
print(y_pred)

# Geriye Dönük Eleme (Backward Elimination)

# sabit terimi ekleme
X = np.append(arr=np.ones((len(sonveriler), 1)).astype(int), values=sonveriler.iloc[:, :-1].values, axis=1)
X_l = sonveriler.iloc[:, [0, 1, 2, 3, 4, 5]].values

# OLS modeli 
r_ols = sm.OLS(endog=sonveriler.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print("\nGeriye Dönük Eleme Sonucu:")
print(r.summary())

# Özelliklerin sayısını azaltma
sonveriler = sonveriler.iloc[:, 1:]

X = np.append(arr=np.ones((len(sonveriler), 1)).astype(int), values=sonveriler.iloc[:, :-1].values, axis=1)
X_l = sonveriler.iloc[:, [0, 1, 2, 3, 4]].values
r_ols = sm.OLS(endog=sonveriler.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print("\nGeriye Dönük Eleme Sonucu (Güncellenmiş):")
print(r.summary())

x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print("\nGüncellenmiş Tahmin Edilen Değerler:")
print(y_pred)
