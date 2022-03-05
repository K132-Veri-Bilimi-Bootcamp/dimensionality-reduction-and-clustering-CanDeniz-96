#!/usr/bin/env python
# coding: utf-8

# # ÖDEV 1: PCA yardımı ile Classification,
# 
# Bu ödevde "Credit Risk Prediction" veri setini kullanacağız. Amacımız, verinin boyut sayısını düşürerek olabildiğince yüksek accuracy değerini alabilmek. Aşağıda verinin okunma ve temizlenme kısmını hazırlayıp vereceğim. Devamında ise yapmanız gerekenler:
# 
# 1. PCA kullanarak verinin boyutunu düşürmek
#     * Önce explained varience ratio değerini inceleyerek veriyi kaç boyuta düşürebileceğini kontrol et.
#     * Daha sonra farklı boyutlarda denemeler yaparak boyutu düşürülmüş verileri elde et.
# 2. Classification modellerini dene
#     * Logistic Regression
#     * Random Forest
#     * ve eğer istersen herhangi bir modelle daha
# 
# İsteğe bağlı olarak, verinin boyutunu düşürmek için diğer yöntemleri de kullanıp en yüksek accuracy değerini almayı deneyebilirsin.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df: pd.DataFrame = pd.read_csv('./credit_risk_dataset.csv')


# In[ ]:


#print(df.isnull().sum())


# In[ ]:


# Null değerleri sütun ortalaması ile dolduruyoruz
df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.describe().T


# In[ ]:


# Outlier temizliği
df = df[df['person_age']<=100]
df = df[df['person_emp_length'] <= 60]
df = df[df['person_income']<=4e6]


# In[ ]:


# Kategorik verileri alıyoruz ve one hot encoding haline getiriyoruz
cat_cols = pd.DataFrame(df[df.select_dtypes(include=['object']).columns])
cat_cols.columns


# In[ ]:


encoded_cat_cols = pd.get_dummies(cat_cols)
df.drop(df.select_dtypes(include=['object']).columns, axis=1,inplace=True)
df = pd.concat([df,encoded_cat_cols], axis=1)


# In[ ]:


X = df.drop('loan_status', axis=1).values
y = df['loan_status'].values


# In[ ]:


# Verileri train ve test olarak ikiye ayırıyoruz

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(1, test_size=0.1)
train_idx, test_idx = next(split.split(X, y))
train_x = X[train_idx]
test_x = X[test_idx]

train_y = y[train_idx]
test_y = y[test_idx]


# ## Kolay gelsin!

# In[ ]:



# import PCA to initialise pca object.

from sklearn.decomposition import PCA



pca =PCA()
x_transformed = pca.fit_transform(X)
print(pca.components_)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))



# Components was defined as 2 because the loss is nearly 1.

pca = PCA(n_components=2)
x2 = pca.fit_transform(X)
print(np.cumsum(pca.explained_variance_ratio_))

print(x2.shape)

# ## Classifications:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

logistic_regression = LogisticRegression()
logistic_regression.fit(train_x, train_y)
prediction = logistic_regression.predict(test_x)
print(prediction)

print(accuracy_score(prediction, test_y))

print(confusion_matrix(prediction, test_y))

# #### Random Forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=300)
clf.fit(train_x, train_y)
tre_prediction = clf.predict(test_x)
print(tre_prediction)
print(accuracy_score(tre_prediction, test_y))
