""" 使用一般的机器学习模型，对综合精度进行 leave one trial out 的交叉验证（使用包）"""

""" 使用一般的机器学习模型进行 "综合分类" 的评估"""

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
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

sub_num = 1

data_all = []
label_all = []

def pca_decomposition(dataset, scaler_judge = False):
    if scaler_judge == True:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
        dataset = scaler.transform(dataset)
    pca = PCA(n_components=8)
    newX = pca.fit_transform(dataset)
    print(pca.explained_variance_ratio_)  # 输出所占比例
    dataset_pca = newX
    return dataset_pca


for sub_index in range(16,16+sub_num):
    path1 = './Feature_2/data'+str(sub_index)+'3.csv'
    path2 = './Feature_2/label'+str(sub_index)+'3.csv'

    dataset = pd.read_csv(path1, header=None)
    label = pd.read_csv(path2, header=None)
    dataset = np.array(dataset)
    label = np.array(label)
    label = label.astype(int)

    data_all.append(dataset)
    label_all.append(label)

data_all = np.array(data_all)
data_all = data_all.reshape(-1,dataset.shape[1])
label_all = np.array(label_all)
label_all = label_all.reshape(-1,5)

dataset = data_all
label = label_all


# 打乱数据，为各个训练器准备训练数据
data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
data_pool[:, 0:dataset.shape[1]] = dataset
data_pool[:, dataset.shape[1]:] = label
np.random.seed(0) #控制伪随机，每次随机划分数据都是相同的
np.random.shuffle(data_pool)
dataset = data_pool[:, 0:dataset.shape[1]]
label = data_pool[:, dataset.shape[1]:]
label = label.astype(int)

#归一化
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
# dataset = scaler.transform(dataset)
# dataset = pca_decomposition(dataset)

""" 构造分类器的集合(这是经过网格调优之后的)"""
knn = KNeighborsClassifier(n_neighbors =  10)                #KNN Modeling
clf = svm.SVC(kernel='rbf', C=10)
rf = RandomForestClassifier(n_estimators=20)       #RF Modeling
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]


# model_scorelist = []  # 保存训练集交叉验证得到的精度数据
# #归一化
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
# dataset = scaler.transform(dataset)
# for model_index in range(4):
#     model = selected_model[model_index]
#     trail_label = label[:,3]
#     mixture_label_train = label[:, 4]
#     logo = LeaveOneGroupOut()
#     scores = cross_val_score(model, dataset, mixture_label_train, groups=trail_label, cv=logo, scoring='precision_macro')
#     print(np.mean(scores))
#     model_scorelist.append(scores)  # 记录不同的模型的交叉验证的准确率
# #print(model_scorelist)

#
# 手动实现留一验证

for model_index in range(4):
    score_list = []
    for i in range(10):
        j = i
        test_index = label[:,3] ==i
        test_index_1 = label[:,3] ==j
        for m in range(test_index_1.shape[0]):
            test_index[m] = test_index[m] or test_index_1[m]
        train_index = ~ test_index
        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        train_label = train_label[:, 4] #由label的类型，确定是哪一种分类任务1手势 2 力 4综合
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]
        test_label = test_label[:, 4]

        model = selected_model[model_index]

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        test_dataset = scaler.transform(test_dataset)


        model.fit(train_dataset, train_label)
        predicted_label = model.predict(test_dataset)
        a_4 = sm.accuracy_score(test_label, predicted_label)

        #a_4 = sm.precision_score(test_label, predicted_label, average='macro')
        #score = model.score(test_dataset, test_label)
        score_list.append(a_4)



    mean_acc = np.mean(score_list)
    score_list.append(mean_acc)#最后一列增加平均精度
    print(score_list)
