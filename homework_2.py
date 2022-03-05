#!/usr/bin/env python
# coding: utf-8

# # Ödev 2: K-Means ve California Housing Dataset
# 
# Bu ödevde California housing veri setini, Kmeans algoritmasını kullanarak ekonomik segmentlere ayırmanız gerekmektedir. Alt görevler:
# 
# 1. Verisetini kullanarak farklı cluster sayılarında KMeans clusterları eğit (2-8 arası iyi bir seçim)
# 2. Her bir KMeans için silhouette skorunu çıkar ve görselleştir. Eğer istersen Silhouette grafiğini de çıkarabilirsin.
# 3. Farklı cluster sayılarının, segmentasyona etkilerini incele. Örneğin cluster sayısı 2 olduğunda haritayı 2'ye bölüp yukarıdakiler ve aşağıdakiler şeklinde ayırıyor, 3 olduğunda ise yukarıdakiler, aşağıdakiler ve zenginler gibi ayırıyor. Bol bol keşfet!

# In[81]:


from cProfile import label
from sklearn.datasets import fetch_california_housing

dataset  = fetch_california_housing(as_frame = True)


# In[82]:


X = dataset.data
X


# In[83]:


import matplotlib.pyplot as plt

plt.scatter(X['Longitude'],X['Latitude'])


# In[84]:


X = X.loc[:, ["MedInc", "Latitude", "Longitude"]]


# In[85]:


X.head()


# ## Kolay gelsin!

# In[ ]:




#print(dataset.target)

# import KMeans and Silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
for k in range(2,9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sil_score = silhouette_score(X, kmeans.labels_)
    sil_sample = silhouette_samples(X, kmeans.labels_)
    print(f"silhouette_score for {k}: {sil_score}")
    # To understand whether the cluster is good.
    # To do this, we look how many points are smaller than silhouette_score.
    points_count = len([i for i in sil_sample if i < sil_score])
    print(points_count, "/", len(sil_sample))

