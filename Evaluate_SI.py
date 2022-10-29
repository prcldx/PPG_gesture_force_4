"""本文件的目的是，对特征空间的分布情况进行描述，使用指标SI，最终目标是出一个热力图
SI 值越小，说明两个类之间的距离越大"""
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
from Tool_Functionset import *


def SI_evaluate(train_data, test_data):
    eps = 0.001
    sigma = np.std(train_data, axis=0, ddof=1)
    max_sigma = max(sigma)
    train_cd = np.mean(train_data, axis=0)
    test_cd = np.mean(test_data, axis=0)
    E_distance = np.sqrt(sum((train_cd- test_cd)**2))
    si = max_sigma/(E_distance+eps)
    #si = 1/E_distance
    return si


sub_num = 9
SI_matrix = np.zeros((4, 4))
for sub_index in range(1,1+sub_num):
    path1 = './Feature_2/data'+str(sub_index)+'3.csv'
    path2 = './Feature_2/label'+str(sub_index)+'3.csv'
    dataset = pd.read_csv(path1, header=None)
    dataset = np.array(dataset)
    #可以选择使用不同的信号来测试二者的SI
    #dataset = dataset[:,range(12,18)]
    label = pd.read_csv(path2, header=None)
    label = np.array(label)
    label = label.astype(int)

    # 最大最小归一化
    # scaler = preprocessing.StandardScaler().fit(dataset)
    # #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
    # dataset = scaler.transform(dataset)
    #循环进行求SI值

    for train_index in range(4):
        for test_index in range(4):
            #取出MVC70的数据作为训练集
            train_data = dataset[label[:,1]==train_index,:]
            train_label = label[label[:, 1] == train_index]
            train_data = train_data[train_label[:, 2] == 2, :]
            train_label = train_label[train_label[:, 2] == 2, :]

            #取出MVC10的数据作为测试集
            test_data = dataset[label[:,1]==test_index,:]
            test_label = label[label[:,1]==test_index,:]
            test_data = test_data[test_label[:,2] == 0, :]
            test_label = test_label[test_label[:,2] == 0,:]

            #调用SI评估函数
            si = SI_evaluate(train_data, test_data)
            SI_matrix[train_index, test_index] = SI_matrix[train_index, test_index]+si
            print(1)

SI_matrix = SI_matrix/sub_num
print(1)
labels_name = ['TIP', 'TMP', 'TRP', 'KP']
plt.figure(figsize=(8,8))
plt.imshow(SI_matrix, interpolation='nearest', cmap=plt.cm.Greens)
cb1 = plt.colorbar(fraction=0.04)
tick_locator = ticker.MaxNLocator(nbins=6)
cb1.locator = tick_locator
cb1.update_ticks()
num_local = np.array(range(len(labels_name)))
plt.xticks(num_local, labels_name, fontsize=10, rotation=60)  # 将标签印在x轴坐标上
plt.yticks(num_local, labels_name, fontsize=10, rotation=60)  # 将标签印在y轴坐标上
plt.ylabel('70% MVC Force Level', fontsize=12)
plt.xlabel('10% MVC Force level', fontsize=12)
plt.show()
