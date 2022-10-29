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

path1 = './Feature/data31.csv'
path2 = './Feature/labe31.csv'

dataset = pd.read_csv(path1, header= None)
label = pd.read_csv(path2, header = None)
dataset = np.array(dataset)
label = np.array(label)
label = label.astype(int)


# 打乱数据，为各个训练器准备训练数据
data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
data_pool[:, 0:dataset.shape[1]] = dataset
data_pool[:, dataset.shape[1]:] = label
np.random.seed(0) #控制伪随机，每次随机划分数据都是相同的
np.random.shuffle(data_pool)
dataset = data_pool[:, 0:dataset.shape[1]]
label = data_pool[:, dataset.shape[1]:]
label = label.astype(int)


confusion_matrix_list = []
acc_list = []

fold = 5 #总共进行5次验证结果
kf = KFold(n_splits=fold)
for train_index, test_index in kf.split(dataset):
    train_dataset = dataset[train_index,:]
    train_label = label[train_index,:]
    test_dataset = dataset[test_index,:]
    test_label = label[test_index,:]
    model_a = RandomForestClassifier(n_estimators=20, max_depth=20)

    #进行归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)

    #设置model A 的标签
    train_label_a = train_label[:,1]
    test_label_a = test_label[:,1]
    model_a.fit(train_dataset, train_label_a)
    #
    # 构建各个手势单独的力分类器
    model_b_list = []
    for j in range(4):
        #rf = RandomForestClassifier(n_estimators=16, max_depth=8)
        model_b_list.append(RandomForestClassifier(n_estimators=16, max_depth=8))
        #model_b_list.append(LinearDiscriminantAnalysis())

    for j in range(4):
        #构造model_b训练器的 训练集和测试集
        train_label_b = train_label[train_label[:,1]==j]
        train_label_b = train_label_b[:,2]
        train_dataset_b = train_dataset[train_label[:,1]==j]

        #为手势和力取出最后结合的标签,为最后计算准确率做准备
        final_train_label = train_label[train_label[:,1] == j][:,4]
        final_test_label = test_label[test_label[:,1] == j][:,4]

        #为力分类器进行训练
        model_b_list[j].fit(train_dataset_b, train_label_b)

    # 测试这个串行分类器的指标
    # 对test_dataset进行归一化
    test_dataset = scaler.transform(test_dataset)

    #
    predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
    test_score = model_a.score(test_dataset, test_label_a)

    # 基于predict_a的结果继续进行分类
    confusion_array = np.zeros((15, 15))

    #对每个手势中的力分别进行分类
    for j in range(4):
        test_dataset_b = test_dataset[predict_a == j]
        final_test_label = test_label[predict_a == j]
        force_label = final_test_label[:, 2]  # 构建力标签
        final_test_label = final_test_label[:, 4]  # 构建最终的评价标签,这个是真值

        # 预测力
        predict_b = model_b_list[j].predict(test_dataset_b)

        # 评估最后的预测结果
        final_predict = predict_a[predict_a == j] * 3 + predict_b

        for m in range(12):
            for n in range(j * 3, (j + 1) * 3):
                judge_list1 = final_test_label == m
                judge_list2 = final_predict == n

                judge_list2 = judge_list2[judge_list1]
                judge_list2 = list(judge_list2)

                confusion_array[m, n] = judge_list2.count(True)

    # 统计准确率
    total = np.sum(confusion_array)
    # 统计对角线的值
    temp = 0
    for index in range(15):
        temp = temp + confusion_array[index, index]
    acc = temp / total
    # print(acc)
    acc_list.append(acc)
    confusion_matrix_list.append(confusion_array)  # 统计4次交叉验证的混淆矩阵

    # 可视化（混淆矩阵）
    labelstr_list = []
    # for i in range(15):
    #     labelstr_list.append(str(i))

    # plot_confusion_matrix(confusion_array, labelstr_list, "Gesture and Force Classification")
    # plt.show()

print("总体的分类精度为：%.4f%%\n" % (np.mean(acc_list) * 100))
print(acc_list)

