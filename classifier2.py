""" 使用一般的机器学习模型进行 "综合分类" 的评估 k-FOLD Validation"""

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


path1 = './Feature/data12.csv'
path2 = './Feature/label12.csv'

dataset = pd.read_csv(path1, header= None)
label = pd.read_csv(path2, header = None)
dataset = np.array(dataset)
label = np.array(label)
label = label.astype(int)


# 打乱数据，为各个训练器准备训练数据
data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
data_pool[:, 0:dataset.shape[1]] = dataset
data_pool[:, dataset.shape[1]:] = label
np.random.seed(0)
np.random.shuffle(data_pool)
dataset = data_pool[:, 0:dataset.shape[1]]
label = data_pool[:, dataset.shape[1]:]
label = label.astype(int)

""" 构造分类器的集合"""
knn = KNeighborsClassifier()                #KNN Modeling
clf = svm.SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators=20)       #RF Modeling
#lda = justatest()
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]

for model_index in range(4):
    fold = 5  # 总共进行5次验证结果
    kf = KFold(n_splits=fold)
    score_list = []
    for train_index, test_index in kf.split(dataset):
        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        train_label = train_label[:, 4]
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]
        test_label = test_label[:, 4]

        model = selected_model[model_index]

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        test_dataset = scaler.transform(test_dataset)

        model.fit(train_dataset, train_label)
        score = model.score(test_dataset, test_label)
        score_list.append(score)
        predicted_label = model.predict(test_dataset)

        m = sm.confusion_matrix(test_label, predicted_label)
        r = sm.classification_report(test_label, predicted_label)
        #print(r)
        #a_1 = sm.average_precision_score(test_label, predicted_label)
        a_2 = sm.accuracy_score(test_label, predicted_label)

        a_3 = sm.recall_score(test_label, predicted_label, average=None)
        a_4 = sm.recall_score(test_label, predicted_label, average='macro')

        #print('混淆矩阵为\n',m)
        #print('分类报告为\n',r)
        #r_list = list(r)
        #r_pd = pd.DataFrame(r)

    mean_acc = np.mean(score_list)
    score_list.append(mean_acc)#最后一列增加平均精度
    print(score_list)




