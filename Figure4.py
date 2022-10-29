""" 不区分人，将总体作为研究对象"""

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


sub_num = 2
sub_score_array = []
for sub_index in range(1,sub_num+1):

    path1 = './Feature/data' + str(sub_index) + '2.csv'
    path2 = './Feature/label' + str(sub_index) + '2.csv'

    dataset = pd.read_csv(path1, header=None)
    label = pd.read_csv(path2, header=None)
    dataset = np.array(dataset)
    label = np.array(label)
    label = label.astype(int)

    # 打乱数据，为各个训练器准备训练数据
    data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
    data_pool[:, 0:dataset.shape[1]] = dataset
    data_pool[:, dataset.shape[1]:] = label
    np.random.seed(0)  # 控制随机数产生的种子
    np.random.shuffle(data_pool)
    dataset = data_pool[:, 0:dataset.shape[1]]
    label = data_pool[:, dataset.shape[1]:]
    label = label.astype(int)

    # 构造分类器的集合
    knn = KNeighborsClassifier()  # KNN Modeling
    clf = svm.SVC(kernel='rbf', C=10)
    rf = RandomForestClassifier(n_estimators=20)  # RF Modeling
    lda = LinearDiscriminantAnalysis()
    selected_model = [rf, lda, knn, clf]
    force_score_classifier_array = np.zeros((4, 10))
    gesture_score_classifier_array = np.zeros((4, 10))

    # 循环选择模型
    model_score_array = []
    for model_index in range(4):
        candidate_model_a = selected_model[model_index]  # 选择只针对手势分类的分类器
        candidate_model_b = selected_model[model_index]  # 选择只针对力分类的分类器
        candidate_model_c = selected_model[model_index]  # 选择总共的分类器


        gesture_score, force_score, total_score = lot_validate(dataset, label, candidate_model_a, candidate_model_b, candidate_model_c)
        score_list = [gesture_score, force_score, total_score]
        model_score_array.append(score_list)
    model_score_array = np.array(model_score_array)
    sub_score_array.append(model_score_array)

sub_score_array = np.array(sub_score_array)#0维：subject 1维度:model 2维：3个score
print(1)

