
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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path1 = './Feature/data22.csv'
path2 = './Feature/label22.csv'

dataset = pd.read_csv(path1, header= None)
label = pd.read_csv(path2, header = None)
dataset = np.array(dataset)
label = np.array(label)
label = label.astype(int)

#取出一个手势中的3个力用PCA进行空间上的显示
dataset = dataset[label[:,1]==0]
label = label[label[:,1]==0]
label = label[:,2] #取出手势作为标签

#标准化

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
dataset = scaler.transform(dataset)

#pca降维
pca = PCA(n_components=3)
newX = pca.fit_transform(dataset)
dataset = newX

X=dataset
y=label

#画图
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

# y_marker = ['o','^','*','.','p']
# color_name = ['red','green','purple','c','y']
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# plt.show()

#绘制二维散点图
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

#绘制三维散点图
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

#y_marker = ['o','^','*','.','p']
y_marker = ['^','*','.']
#color_name = ['red','green','purple','c','y']
color_name = ['red','green','purple']

for i in range(3):
    ax.scatter(X[label==i, 0], X[label==i, 1], X[label==i, 2], c=color_name[i], marker=y_marker[i])



ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()
