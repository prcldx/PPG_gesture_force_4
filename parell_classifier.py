""" 使用一般的机器学习模型，对综合精度进行 leave one trial out 的交叉验证,设计并联的模型"""

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
from ReliefF import ReliefF
from sklearn.neighbors import NeighborhoodComponentsAnalysis

import warnings
from relefF2 import *
warnings.filterwarnings("ignore")

sub_num = 1

def pca_decomposition(dataset, scaler_judge = False):
    if scaler_judge == True:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
        dataset = scaler.transform(dataset)
    pca = PCA(n_components=8)
    newX = pca.fit_transform(dataset)
    print(pca.explained_variance_ratio_)  # 输出所占比例
    dataset_pca = newX
    return dataset_pca
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
clf2 = svm.SVC(kernel='rbf', C=10)  #为力分类器构建模型


#定义用于降维特征选择的类
class Feature_Selection():
    def __init__(self,dataset,label):
        self.dataset = dataset
        self.label = label

    def basic_NCA(self, component_num, label_target):
        # Normalization
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.dataset)
        self.dataset = scaler.transform(self.dataset)
        nca = NeighborhoodComponentsAnalysis(n_components=component_num)
        scaler = nca.fit(self.dataset, label_target)
        dataset_new = scaler.transform(self.dataset)
        return dataset_new

    def basic_ReliefF(self,label_target, component_num):
        # Normalization
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.dataset)
        self.dataset = scaler.transform(self.dataset)
        fs = ReliefF(n_neighbors=40, n_features_to_keep=component_num)
        dataset_new = fs.fit_transform(self.dataset, label_target)
        return dataset_new

    def NCA_select(self, component_num):
        # label
        label_g = self.label[:, 1]
        label_f = self.label[:, 2]

        # NCA gesture
        dataset_g = self.basic_NCA(component_num, label_g)
        dataset_f = self.basic_NCA(component_num, label_f)
        return dataset_g, dataset_f

    def ReliefF_select(self, component_num):
        # label
        label_g = self.label[:, 1]
        label_f = self.label[:, 2]
        # NCA gesture
        dataset_g = self.basic_ReliefF(label_target=label_g, component_num = component_num)
        dataset_f = self.basic_ReliefF(label_target=label_f, component_num = component_num)
        return dataset_g, dataset_f

# f = Feature_Selection(dataset, label)
# dataset_g, dataset_f = f.ReliefF_select(component_num=10)

# rel = ReliefF_2()
# rel.fit(dataset, label[:,1])
# print(rel.top_features)
#
#
# rel = ReliefF_2()
# rel.fit(dataset, label[:,2])
# print(rel.top_features)

label_g = label[:,1].reshape(-1,1)
dataset_g = MultiReliefF(n_neighbors=10, n_features_to_keep=10, n_selected=100).fit(dataset, label_g)

label_f = label[:,2].reshape(-1,1)
dataset_f = MultiReliefF(n_neighbors=10, n_features_to_keep=10, n_selected=100).fit(dataset, label_f)



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


        train_dataset_g = dataset_g[train_index, :]
        train_dataset_f = dataset_f[train_index, :]

        train_label = label[train_index, :]
        train_label_f = train_label[:, 2] #由label的类型，确定是哪一种分类任务1手势 2 力 4综合
        train_label_g = train_label[:, 1]

        test_dataset_g = dataset_g[test_index, :]
        test_dataset_f = dataset_f[test_index, :]

        test_label = label[test_index, :]
        test_label_f = test_label[:, 2]
        test_label_g = test_label[:, 1]
        test_label_mix = test_label[:, 4]

        model = selected_model[model_index]
        model_f = clf2

        # 归一化
        scaler_g = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset_g)
        train_dataset_g = scaler_g.transform(train_dataset_g)
        test_dataset_g = scaler_g.transform(test_dataset_g)

        scaler_f = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset_f)
        train_dataset_f = scaler_f.transform(train_dataset_g)
        test_dataset_f = scaler_f.transform(test_dataset_f)

        model.fit(train_dataset_g, train_label_g)
        model_f.fit(train_dataset_f, train_label_f)

        predicted_label_g = model.predict(test_dataset_g)
        predicted_label_f = model_f.predict(test_dataset_f)
        predicted_label = predicted_label_g*3 + predicted_label_f
        a_4 = sm.accuracy_score(test_label_mix, predicted_label)
        #a_4 = sm.precision_score(test_label_mix, predicted_label)
        #score = model.score(test_dataset, test_label)
        score_list.append(a_4)



    mean_acc = np.mean(score_list)
    score_list.append(mean_acc)#最后一列增加平均精度
    print(score_list)
