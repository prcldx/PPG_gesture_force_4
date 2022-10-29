""" 根据多数文献的选择，使用K-fold Validation 使用新的串联分类器"""
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

def read_feature(sub_name, sub_num = 1,):
    data_all = []
    label_all = []
    for sub_index in range(sub_name, sub_name + sub_num):
        path1 = './Feature/data' + str(sub_index) + '3.csv'
        path2 = './Feature/label' + str(sub_index) + '3.csv'

        dataset = pd.read_csv(path1, header=None)
        label = pd.read_csv(path2, header=None)
        dataset = np.array(dataset)
        label = np.array(label)
        label = label.astype(int)

        data_all.append(dataset)
        label_all.append(label)

    data_all = np.array(data_all)
    data_all = data_all.reshape(-1, dataset.shape[1])
    label_all = np.array(label_all)
    label_all = label_all.reshape(-1, 5)

    dataset = data_all
    label = label_all

    # 打乱数据，为各个训练器准备训练数据
    data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
    data_pool[:, 0:dataset.shape[1]] = dataset
    data_pool[:, dataset.shape[1]:] = label
    np.random.seed(0)  # 控制伪随机，每次随机划分数据都是相同的
    np.random.shuffle(data_pool)
    dataset = data_pool[:, 0:dataset.shape[1]]
    label = data_pool[:, dataset.shape[1]:]
    label = label.astype(int)
    return dataset, label
dataset, label = read_feature(sub_name=9)

""" 构造分类器的集合(这是经过网格调优之后的)"""
knn = KNeighborsClassifier(n_neighbors =  10)                #KNN Modeling
clf = svm.SVC(kernel='rbf', C=10)
rf = RandomForestClassifier(n_estimators=20)       #RF Modeling
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]

# 进行lof验证
score_list = []
for i in range(10):
    j = i
    test_index = label[:, 3] == i
    test_index_1 = label[:, 3] == j
    for m in range(test_index_1.shape[0]):
        test_index[m] = test_index[m] or test_index_1[m]
    train_index = ~ test_index

    train_dataset = dataset[train_index, :]
    train_label = label[train_index, :]
    test_dataset = dataset[test_index, :]
    test_label = label[test_index, :]

    model_a = svm.SVC(kernel='rbf', C=10)
    # 进行归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)

    # 设置model A 的标签
    train_label_a = train_label[:, 1]
    test_label_a = test_label[:, 1]
    model_a.fit(train_dataset, train_label_a)

    # 构建各个手势单独的力分类器
    model_b_list = []
    for j in range(4):
        model_b_list.append(svm.SVC(kernel='rbf', C=10))

    for j in range(4):
        #构造model_b训练器的 训练集和测试集
        train_label_b = train_label[train_label[:,1]==j]
        train_label_b = train_label_b[:,2]
        train_dataset_b = train_dataset[train_label[:,1]==j]
        #为力分类器进行训练
        model_b_list[j].fit(train_dataset_b, train_label_b)

# 测试这个串行分类器的指标
    # 对test_dataset进行归一化
    test_dataset = scaler.transform(test_dataset)

    predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
    test_score = model_a.score(test_dataset, test_label_a)

    #对每个手势中的力分别进行分类
    final_predict_list = []
    final_true_list = []
    for j in range(4):
        test_dataset_b = test_dataset[predict_a == j]
        final_test_label = test_label[predict_a == j]
        force_label = final_test_label[:, 2]  # 构建力标签
        final_test_label = final_test_label[:, 4]  # 构建最终的评价标签,这个是真值

        # 预测力
        predict_b = model_b_list[j].predict(test_dataset_b)

        # 评估最后的预测结果
        final_predict = predict_a[predict_a == j] * 3 + predict_b

        final_predict_list.extend(final_predict)
        final_true_list.extend(final_test_label)

    s1 = sm.accuracy_score(final_true_list, final_predict_list)
    score_list.append(s1)
mean_precision = np.mean(score_list)
print(score_list)
print(mean_precision)




