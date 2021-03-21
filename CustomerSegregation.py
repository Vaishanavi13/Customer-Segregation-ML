#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns  #Python library for Visualization


# In[3]:


#importing dataset
dataset = pd.read_csv(r'C:\Users\HP\Downloads\customer-segmentation-dataset\customer-segmentation-dataset\Mall_Customers.csv')

dataset.head(10) #Printing first 10 rows of the dataset
dataset.info()


# In[21]:


#Missing values computation
dataset.isnull().sum()

### Feature sleection for the model
#Considering only 2 features (Annual income and Spending Score)
X= dataset.iloc[:, [3,4]].values


# In[22]:


#KMeans Algorithm to decide the optimum cluster number
#to figure out K for KMeans, using ELBOW Method on KMEANS Calculation

from sklearn.cluster import KMeans
wcss=[]

#assuming the max number of cluster would be 10

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters


# In[23]:


#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#Model Building
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)


# In[26]:


#Visualizing all the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], marker = "*", s = 100, c = '#bd121a', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], marker = "*", s = 100, c = '#f0a618', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], marker = "*", s = 100, c = '#1d881a', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], marker = "*", s = 100, c = '#20536a', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], marker = "*", s = 100, c = '#d51257', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




