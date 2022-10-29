" "
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


def RI_evaluate(train_data, test_data):
    meanval = np.mean(train_data, axis=0)
    newData = train_data - meanval #去均值化
    covMat = np.cov(newData, rowvar=0)#计算协方差矩阵
    train_cd = meanval
    test_cd = np.mean(test_data, axis=0)
    cd_distance = train_cd - test_cd

    cd_distance_T = np.transpose(cd_distance)
    temp1 = np.dot(cd_distance_T, covMat)
    temp2 = np.dot(temp1, cd_distance)
    ri = 0.5*np.sqrt(temp2)
    #cd_distance_T = np.matrix(cd_distance_T)
    return ri


# 基于numpy自己实现PCA
# import numpy as np
#
# meanval = np.mean(X_arr, axis=0)  # 计算原始数据中每一列的均值，axis=0按列取均值
# newData = X_arr - meanval  # 去均值化，每个feature的均值为0
# covMat = np.cov(newData, rowvar=0)  # 计算协方差矩阵，rowvar=0表示数据的每一列代表一个feature
# featValue, featVec = np.linalg.eig(covMat)  # 计算协方差矩阵的特征值和特征向量
# index = np.argsort(featValue)  # 将特征值按从大到小排序，index保留的是对应原featValue中的下标
# n_index = index[-1:-3:-1]  # 取最大的两维特征值在原featValue中的下标
# n_featVec = featVec[:, n_index]  # 取最大的两维特征值对应的特征向量组成变换矩阵
# lowData = np.dot(newData, n_featVec)  # lowData=newData*n_featVec
# highData = np.dot(lowData, n_featVec.T) + meanval  # highData=(lowData*n_featVec.T)+meanval



RI_flag = 1
sub_num = 14
RI_matrix = np.zeros((sub_num,12))
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
    #scaler = preprocessing.StandardScaler().fit(dataset)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
    dataset = scaler.transform(dataset)
    #循环进行求RI值,始终选择70MVC作为训练集
    for gesture_index in range(4):
        for test_index in range(3):
            # 取出第一个手势MVC70的数据作为训练集
            train_data = dataset[label[:, 1] == gesture_index, :]
            train_label = label[label[:, 1] == 0]
            train_data = train_data[train_label[:, 2] == 2, :]
            train_label = train_label[train_label[:, 2] == 2, :]

            # 轮流取出MVC10 40 70 的数据作为测试集
            test_data = dataset[label[:, 1] == gesture_index, :]
            test_label = label[label[:, 1] == gesture_index, :]
            test_data = test_data[test_label[:, 2] == test_index, :]
            test_label = test_label[test_label[:, 2] == test_index, :]

            ri = RI_evaluate(train_data, test_data)
            RI_matrix[sub_index-1,gesture_index*3 + test_index] = ri
a = np.max(RI_matrix, axis=1)
a = a.reshape(-1,1)
RI_matrix = RI_matrix/a
RI_mean = np.mean(RI_matrix, axis=0)
RI_std = np.std(RI_matrix, axis=0, ddof=1)

#
# print(1)
# gesture_name = ['TIP','TMP','TRP','KP']
# labels_name = [ 'TIP10', 'TIP40', 'TIP70', 'TMP10', 'TMP40', 'TMP70', 'TRP10', 'TRP40', 'TRP70', 'KP10', 'KP40', 'KP70' ]
# label_index = []
# if RI_flag==1:
#     plt.rcParams['font.family'] = 'Arial'
#     plt.rcParams.update({'font.size': 6})
#     plt.figure(figsize=(3.5,3), dpi=600)
#     ax = plt.gca()
#     for i in range(0,4):
#         x = np.arange(i,i+3*0.3-0.1,0.3)
#         label_index.extend(list(x))
#         y = RI_mean[i*3:(i+1)*3]
#         error = RI_std[i*3:(i+1)*3]
#         plt.bar(x, y * 100, alpha=0.6, width=0.2, lw=3, label=gesture_name[i])
#         plt.errorbar(x, y * 100, yerr=error * 100, fmt='.', ecolor='black',
#                      elinewidth=0.5, ms=1, mfc='wheat', mec='salmon', capsize=3, capthick = 0.5)
#
#     bwith = 0.5
#     ax.spines['bottom'].set_linewidth(bwith)
#     ax.spines['left'].set_linewidth(bwith)
#     ax.spines['top'].set_linewidth(bwith)
#     ax.spines['right'].set_linewidth(bwith)
#
#     plt.xticks(label_index, labels_name,  rotation=60)
#     plt.xlabel('Gesture set', loc='center')
#     plt.ylabel('Repeated Index ')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('C:\\Users\\Lenovo\\Desktop\\New_Figure\\fri.png')
#     #plt.ylim(0, 10)
#     plt.show()
