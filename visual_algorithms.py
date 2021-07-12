# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:38:42 2021

@author: Bernardo
"""
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sammon import sammon
from fcm import fcm
from fcmfp import fcmfp
from gg import gg
from ggfp import ggfp



algorithm='ggfp'             #algorithm='fcm','fcmfp','gg','ggfp'
dataset='iris'              #dataset='iris','bearing'
nClusters=3




def load_data(dataset):
    
    if dataset=='iris':
        "Iris data"
        iris = datasets.load_iris()
        data = iris.data 
        label_iris = iris.target
        label_gt=label_iris
        
    elif dataset=='bearing':
                
        "Bearing data"
        data = pd.read_csv("bearingdata.csv",header=None)        
        data=data.to_numpy()  
        #import ground truth labels
        label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
        label_bearing=label_bearing.to_numpy().T
        label_bearing=label_bearing.reshape(len(label_bearing))
        
        label_gt=label_bearing #ground truth label

    return data , label_gt

#Load the data and the gt label
data, label_gt =load_data(dataset)
# Number of samples
n = len(data)  # the number of row
d = len(data[0]) #number of features/attributes


"Define Focal point"
#P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
#P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset

#focal point using mean of data
datamean=np.array(np.mean(data,axis=0))
Pvalue=np.array([10])
P=np.append(datamean,Pvalue,axis=0)
P=np.reshape(P,(1,d+1))


w=np.shape(P)[1]     #nº of columns : dimension of P



if algorithm=='fcm':
    
    "#Call and run the fcm algorithm"
    

    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random   
    #Initialize centers of clusters
    #C=np.zeros(nClusters,d)
    C = rng.random((nClusters,d)) 
    
    C, U, labels, obj_fcn = fcm(data,C,nClusters,2)   #FCM(C,nClusters,m)
    
    "Sammon Projection"
    # Run the Sammon projection
    
    sammon_data=np.concatenate((data, C), axis=0)
    #Project data points
    
    (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
    labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
    target = labels2[index] 
    [y,E] = sammon(x,2)
    
    
    
    # Plot
    arr_str_color = ['g','b','y','m','y','k','gray','orange','tab:pink','brown'] 
    arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
    
    for i in range(nClusters):
        #Plot data points
        plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
    #Plot centers
    plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='o')
    
    
    plt.title('Sammon projection of the data - FCM')
    #plt.legend(loc='lower left')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
elif algorithm=='fcmfp':
    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random
    
    C = rng.random((nClusters,w))    #Initialize centers of clusters
    
    # C = np.random.rand(nClusters,w)    #Initialize centers of clusters
    
    C, U, labels, obj_fcn, projCenter = fcmfp(data,P, C,nClusters,0.2,2)   #FCMFP(C,nClusters,zeta,m)

    "Sammon Projection"
    # Run the Sammon projection
    original_data=np.delete(data,-1,axis=1)
    
    sammon_data=np.concatenate((data, projCenter), axis=0)
    #Project data points
    
    (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
    labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
    target = labels2[index] 
    [y,E] = sammon(x,2)
    
    
    
    # Plot
    arr_str_color = ['g','b','y','m','y','k','gray','orange','tab:pink','brown'] 
    arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
    
    for i in range(nClusters):
        #Plot data points
        plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
    #Plot centers
    plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='o')
    
    
    plt.title('FCMFP - Sammon projection of the data ')
    # plt.legend(loc='lower left')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()


elif algorithm=='gg':
    U, C, A,labels, dist, obj_fcn,Pi ,M=gg(data,nClusters,2,init='fcm')  #gg(data,nClusters,m,init='fcm','kmeans++')

    "Sammon Projection"
    # Run the Sammon projection
    
    sammon_data=np.concatenate((data, C), axis=0)
    #Project data points
    
    (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
    labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
    target = labels2[index] 
    [y,E] = sammon(x,2)
    
    
    
    # Plot
    arr_str_color = ['g','b','y','m','y','k','gray','orange','tab:pink','brown'] 
    arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
    
    for i in range(nClusters):
        #Plot data points
        plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
    #Plot centers
    plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='o')
    
    
    plt.title('GG - Sammon projection of the data ')
    # plt.legend(loc='lower left')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()

elif algorithm=='ggfp':    
    #  GGFP(data,P,nClusters,zeta,m,init='fcm','kmeans++')
    C, U, labels, obj_fcn,dist,dist_FP,M,Pi,dist_C,projCenter = ggfp(data,P,nClusters,0.1,2,init='fcm')  
    # "Sammon Projection"
    # Run the Sammon projection
    original_data=np.delete(data,-1,axis=1)
    
    sammon_data=np.concatenate((data, projCenter), axis=0)
    #Project data points
    
    (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
    labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
    target = labels2[index] 
    [y,E] = sammon(x,2)
    
    
    
    # Plot
    arr_str_color = ['g','b','y','m','y','k','gray','orange','tab:pink','brown'] 
    arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
    
    for i in range(nClusters):
        #Plot data points
        plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
    #Plot centers
    plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='o')
    
    
    plt.title("GGFP- Sammon projection of dataset")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    # plt.legend(loc='lower left')
    plt.show()
    
    
    
