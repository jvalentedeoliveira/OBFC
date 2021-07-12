# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:36:13 2021
From:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

@author: Bernardo
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets

nClusters=3

iris = datasets.load_iris()
X = iris.data 
n = len(X)  # the number of row
d = len(X[0]) #number of features/attributes


#initialization: kmeans++ or random
kmeans = KMeans(nClusters,init='k-means++', random_state=0).fit(X)

C=kmeans.cluster_centers_
label_iris = iris.target





#print('Kmeans++ Adjusted Rand Index :', adjusted_rand_score(label_iris, kmeans.predict(X)))

         

