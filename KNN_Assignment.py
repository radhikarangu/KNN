# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:14:50 2020

@author: RADHIKA
"""
###########Glass Assignment######################
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
glass=pd.read_csv("D:\\ExcelR Data\\Assignments\\KNN\\glass.csv")
glass.head
glass.columns
glass.shape
glass_ip=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
glass_op=['Type']
Xtrain,Xtest,Ytrain,Ytest=train_test_split(glass[glass_ip],glass[glass_op],test_size=0.2)

knn=KNC(n_neighbors=3)
knn.fit(Xtrain,Ytrain.values.ravel())
#train_acc=acc(knn.predict(Xtrain),Ytrain.values.ravel())
#test_acc=acc(knn.predict(Xtest),Ytest.values.ravel())
##For k=3
train_acc=np.mean(knn.predict(Xtrain)==Ytrain.values.ravel())####80%
test_acc=np.mean(knn.predict(Xtest)==Ytest.values.ravel())###77%
print(train_acc)
print(test_acc)

###For k=5
knn=KNC(n_neighbors=5)
knn.fit(Xtrain,Ytrain.values.ravel())
train_acc=np.mean(knn.predict(Xtrain)==Ytrain.values.ravel())####74%
test_acc=np.mean(knn.predict(Xtest)==Ytest.values.ravel())###70%
print(train_acc)
print(test_acc)
import matplotlib.pyplot as plt

plt.plot(train_acc, test_acc)


#################Zoo Assignment#########################

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
zoo=pd.read_csv("D:\\ExcelR Data\\Assignments\\KNN\\Zoo.csv")
zoo.shape
zoo.columns
zoo_ip=zoo.iloc[:,1:-1]
zoo_op=zoo.iloc[:,-1]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(zoo.iloc[:,1:-1],zoo.iloc[:,-1],test_size=0.2)
knn=KNC(n_neighbors=3)
knn.fit(Xtrain,Ytrain.values.ravel())
train_acc=np.mean(knn.predict(Xtrain)==Ytrain.values.ravel())
test_acc=np.mean(knn.predict(Xtest)==Ytest.values.ravel())
print(train_acc)###0.975
print(test_acc)###0.9523
