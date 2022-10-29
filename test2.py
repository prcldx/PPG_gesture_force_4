import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import pandas as pd

from User_modify import *
from Tool_Dataprocess import produce_sample_label
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sm
import numpy as np
from Tool_Visualization import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

#双层列表写入文件

# #第一种方法，每一项用空格隔开，一个列表是一行写入文件
# data =[ ['a','b','c'],['a','b','c'],['a','b','c']]
# with open("data.txt","w") as f:                                                   #设置文件对象
#     for i in data:                                                                 #对于双层列表中的数据
#         i = str(i).strip('[').strip(']').replace(',','').replace('\'','')+'\n'  #将其中每一个列表规范化成字符串
#         f.write(i)                                                                 #写入文件


#第二种方法，直接将每一项都写入文件
# data =[ ('a','±','c'),['a','b','c'],['a','b','c']]
# with open("data.csv","w") as f:                                                   #设置文件对象
#     for i in data:                                                                 #对于双层列表中的数据
#         f.writelines(i)
#         #f.write('\t')
#         f.write('\n')#写入文件


# a = [True, True, True, True, False, False]
# b= [True, False, False,True,False, False]
# c = a and b
# print(c)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

# np.random.seed(5)
#
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
#
#
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
#
# plt.cla()
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)
#
#
# # Reorder the labels to have colors matching the cluster results
# #y = np.choose(y, [1, 2, 0]).astype(float)
# y_marker = ['o','^','*']
# color_name = ['red','green','purple']
# for i in range(X.shape[0]):
#     ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=color_name[int(y[i])], marker=y_marker[int(y[i])])
#
#
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
#
# plt.show()

import itertools

array = [0, 1, 2]
data = np.random.randn(10,3)
print(data)
pailie = list(itertools.permutations(array))  # 要list一下，不然它只是一个对象
for i in pailie:
    k = data[:,i]
    print(k)
