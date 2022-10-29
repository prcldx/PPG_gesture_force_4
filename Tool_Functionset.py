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
import warnings
warnings.filterwarnings("ignore")


def k_fold_validate(dataset, label, candidate_model_a, candidate_model_b):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在k折交叉验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_kfold_array;gesture_score_kfold_array输出为手势和力分别的

    """
    """ 以下是手写的k-fold交叉验证的程序"""
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度
    fold = 10  # 总共进行5次验证结果
    force_score_kfold_list = []
    kf = KFold(n_splits=fold)

    model_index = 0
    for i in range(fold):
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
        model_a = candidate_model_a
        #model_a = RandomForestClassifier(n_estimators=16, max_depth=8)  # 需要改变不同分类器的类型

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        # 对test_dataset进行归一化
        test_dataset = scaler.transform(test_dataset)

        """第一层结构 """
        # 设置model A 的标签
        train_label_a = train_label[:, 1]
        test_label_a = test_label[:, 1]
        model_a.fit(train_dataset, train_label_a)
        #
        predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
        test_score = model_a.score(test_dataset, test_label_a)
        gesture_score_list.append(test_score)  # 把每一次KFold交叉验证的分数记录下来

        """ 第二层结构"""
        # 构建各个手势单独的力分类器
        model_b_list = []
        for j in range(4):
            model_b_list.append(candidate_model_b)
            #model_b_list.append(LinearDiscriminantAnalysis())

        model_index = model_index +1 #引入这个参数目的是每一次交叉验证得到的分类器都是崭新的分类器
        # 训练＋测试结果
        force_score_list = []
        for j in range(4):
            # 构造model_b训练器的 训练集和测试集
            train_label_b = train_label[train_label[:, 1] == j]
            train_label_b = train_label_b[:, 2]
            train_dataset_b = train_dataset[train_label[:, 1] == j]

            # 为力分类器进行训练
            model_b_list[j].fit(train_dataset_b, train_label_b)

            # 为手势和力取出最后结合的标签,为最后计算准确率做准备
            test_label_b = test_label[test_label[:, 1] == j]
            test_label_b = test_label_b[:, 2]
            test_dataset_b = test_dataset[test_label[:, 1] == j]

            predict_b = model_b_list[j].predict(test_dataset_b)
            # 评估最后的预测结果
            test_score = model_b_list[j].score(test_dataset_b, test_label_b)
            force_score_list.append(test_score)
        #mean_acc = np.mean(force_score_list)
        #force_score_list.append(mean_acc)
        force_score_kfold_list.append(force_score_list)

    force_score_kfold_array = np.array(force_score_kfold_list)
    force_score_kfold_array = np.mean(force_score_kfold_array, axis=0)
    gesture_score_kfold_array = np.array(gesture_score_list)
    # print(force_score_kfold_array)
    # print(gesture_score_kfold_array)
    return force_score_kfold_array, gesture_score_kfold_array


def gesture_k_fold_validate(dataset, label, candidate_model_a):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在k折交叉验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_kfold_array;gesture_score_kfold_array

    """
    """ 以下是手写的k-fold交叉验证的程序"""
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度
    fold = 5  # 总共进行5次验证结果
    force_score_kfold_list = []
    kf = KFold(n_splits=fold)

    model_index = 0
    for train_index, test_index in kf.split(dataset):
        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]
        model_a = candidate_model_a
        #model_a = RandomForestClassifier(n_estimators=16, max_depth=8)  # 需要改变不同分类器的类型

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        # 对test_dataset进行归一化
        test_dataset = scaler.transform(test_dataset)

        """第一层结构 """
        # 设置model A 的标签
        train_label_a = train_label[:, 1]
        test_label_a = test_label[:, 1]
        model_a.fit(train_dataset, train_label_a)
        #
        predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
        test_score = model_a.score(test_dataset, test_label_a)
        gesture_score_list.append(test_score)  # 把每一次KFold交叉验证的分数记录下来

    gesture_score_kfold_array = np.array(gesture_score_list)
    # print(force_score_kfold_array)
    # print(gesture_score_kfold_array)
    return  gesture_score_kfold_array


def gesture_lot_validate(dataset, label, candidate_model_a):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在k折交叉验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_kfold_array;gesture_score_kfold_array

    """
    """ 以下是手写的lot交叉验证的程序"""
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度

    for trial_index in range(10):
        train_index = label[:,3] != trial_index
        test_index = label[:,3] == trial_index

        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]
        model_a = candidate_model_a
        # model_a = RandomForestClassifier(n_estimators=16, max_depth=8)  # 需要改变不同分类器的类型

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        # 对test_dataset进行归一化
        test_dataset = scaler.transform(test_dataset)

        # 设置model A 的标签
        train_label_a = train_label[:, 1]
        test_label_a = test_label[:, 1]
        model_a.fit(train_dataset, train_label_a)
        #
        predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
        test_score = model_a.score(test_dataset, test_label_a)
        gesture_score_list.append(test_score)  # 把每一次KFold交叉验证的分数记录下来

    gesture_score_lot_array = np.array(gesture_score_list)

    return gesture_score_lot_array



def gesture_lot_validate_multi_level(train_dataset, train_label, test_dataset, test_label, candidate_model_a):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在k折交叉验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_kfold_array;gesture_score_kfold_array

    """
    """ 以下是手写的lot交叉验证的程序"""
    fold =10
    fold_score = []
    for index in range(fold):
        train_index = train_label[:,3] !=index
        test_index = test_label[:,3] == index
        train_dataset_a = train_dataset[train_index]
        train_label_a = train_label[train_index, 1]
        test_dataset_a = test_dataset[test_index]
        test_label_a = test_label[test_index, 1]
        model_a = candidate_model_a

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset_a)
        train_dataset_a = scaler.transform(train_dataset_a)
        # 对test_dataset进行归一化
        test_dataset_a = scaler.transform(test_dataset_a)

        # 设置model A 的标签
        model_a.fit(train_dataset_a, train_label_a)
        #
        predict_a = model_a.predict(test_dataset_a)  # 给出第一次预测的结果
        test_score = model_a.score(test_dataset_a, test_label_a)
        fold_score.append(test_score)
    mean_score = np.mean(fold_score)

    return mean_score


def lot_validate(dataset, label, candidate_model_a, candidate_model_b, candidate_model_C):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在lot验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_lot_array; gesture_score_lot_array; total_score_lot_array

    """
    """ 以下是手写的k-fold交叉验证的程序"""
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度
    force_score_list = [] #记录每一次力分类的精度
    total_score_list = [] #记录总共的分类精度


    model_index = 0
    fold = 10
    for trial_index in range(fold):
        train_index = label[:,3] != trial_index
        test_index = label[:,3] ==trial_index
        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]

        #建立3种分类的模型
        model_a = candidate_model_a
        model_b = candidate_model_b
        model_c = candidate_model_C

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        # 对test_dataset进行归一化
        test_dataset = scaler.transform(test_dataset)

        """手势分类 """
        # 设置model a 的标签
        train_label_a = train_label[:, 1]
        test_label_a = test_label[:, 1]
        model_a.fit(train_dataset, train_label_a)
        #
        predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
        test_score_a = model_a.score(test_dataset, test_label_a)
        gesture_score_list.append(test_score_a)  # 把lot交叉验证的分数记录下来

        """ 力分类"""
        # 设置model b 的标签
        train_label_b = train_label[:, 2]
        test_label_b = test_label[:, 2]
        model_b.fit(train_dataset, train_label_b)

        #
        predict_b = model_b.predict(test_dataset)  # 给出第一次预测的结果
        test_score_b = model_b.score(test_dataset, test_label_b)
        force_score_list.append(test_score_b)  # 把lot交叉验证的分数记录下来

        """ 总分类 """
        #设置model_c的标签
        train_label_c = train_label[:, 4]
        test_label_c = test_label[:, 4]
        model_c.fit(train_dataset, train_label_c)

        #
        predict_c = model_c.predict(test_dataset)  # 给出第一次预测的结果
        test_score_c = model_c.score(test_dataset, test_label_c)
        total_score_list.append(test_score_c)  # 把lot交叉验证的分数记录下来

    gesture_score = np.mean(gesture_score_list)
    force_score = np.mean(force_score_list)
    total_score = np.mean(total_score_list)


    return gesture_score, force_score, total_score

def k_fold_multiaim_validate(dataset, label, candidate_model_a, candidate_model_b, candidate_model_C, flag):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在lot验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_lot_array; gesture_score_lot_array; total_score_lot_array

    """
    """ 已经改成lot的验证程序了"""
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度
    force_score_list = [] #记录每一次力分类的精度
    total_score_list = [] #记录总共的分类精度

    gesture_recall_list = []
    force_recall_list = [] #记录每一次力分类的精度
    total_recall_list = [] #记录总共的分类精度

    gesture_f1_list = []
    force_f1_list = []
    total_f1_list = []


    gesture_true_label = []
    force_true_label = []

    gesture_predicted_label = []
    force_predicted_label = []


    for i in range(10):
        j = i
        test_index = label[:,3] ==i
        test_index_1 = label[:,3] ==j
        for m in range(test_index_1.shape[0]):
            test_index[m] = test_index[m] or test_index_1[m]
        train_index = ~ test_index
        train_dataset = dataset[train_index, :]
        train_label = label[train_index, :]
        test_dataset = dataset[test_index, :]
        test_label = label[test_index, :]

        #建立3种分类的模型
        model_a = candidate_model_a
        model_b = candidate_model_b
        model_c = candidate_model_C

        # 进行归一化
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
        train_dataset = scaler.transform(train_dataset)
        # 对test_dataset进行归一化
        test_dataset = scaler.transform(test_dataset)

        """手势分类 """
        # 设置model a 的标签
        train_label_a = train_label[:, 1]
        test_label_a = test_label[:, 1]
        model_a.fit(train_dataset, train_label_a)
        #
        predict_a = model_a.predict(test_dataset)  # 给出第一次预测的结果
        #test_score_a = model_a.score(test_dataset, test_label_a)
        test_score_a = sm.precision_score(y_true=test_label_a, y_pred=predict_a, average='macro')
        test_recall_a = sm.recall_score(y_true=test_label_a, y_pred=predict_a, average='macro')
        f1_a = sm.f1_score(y_true=test_label_a, y_pred=predict_a, average='macro')

        gesture_score_list.append(test_score_a)  # 把lot交叉验证的分数记录下来
        gesture_recall_list.append(test_recall_a)
        gesture_f1_list.append(f1_a)

        gesture_true_label.extend(list(test_label_a))
        gesture_predicted_label.extend(list(predict_a))


        """ 力分类"""
        # 设置model b 的标签
        train_label_b = train_label[:, 2]
        test_label_b = test_label[:, 2]
        model_b.fit(train_dataset, train_label_b)

        #
        predict_b = model_b.predict(test_dataset)  # 给出第一次预测的结果
        #test_score_b = model_b.score(test_dataset, test_label_b)
        test_score_b = sm.precision_score(y_true=test_label_b, y_pred=predict_b, average='macro')
        force_score_list.append(test_score_b)  # 把lot交叉验证的分数记录下来

        test_recall_b = sm.recall_score(y_true=test_label_b, y_pred=predict_b, average='macro')
        force_recall_list.append(test_recall_b)

        f1_b = sm.f1_score(y_true=test_label_b, y_pred=predict_b, average='macro')
        force_f1_list.append(f1_b)

        force_true_label.extend(list(test_label_b))
        force_predicted_label.extend(list(predict_b))


        """ 总分类 """
        #设置model_c的标签
        train_label_c = train_label[:, 4]
        test_label_c = test_label[:, 4]
        model_c.fit(train_dataset, train_label_c)

        #
        predict_c = model_c.predict(test_dataset)  # 给出第一次预测的结果
        #test_score_c = model_c.score(test_dataset, test_label_c)
        test_score_c = sm.precision_score(y_true=test_label_c, y_pred=predict_c, average='macro')
        total_score_list.append(test_score_c)  # 把lot交叉验证的分数记录下来

        test_recall_c = sm.recall_score(y_true=test_label_c, y_pred=predict_c, average='macro')
        total_recall_list.append(test_recall_c)

        f1_c = sm.f1_score(y_true=test_label_c, y_pred=predict_c, average='macro')
        total_f1_list.append(f1_c)


    gesture_score = np.mean(gesture_score_list)
    force_score = np.mean(force_score_list)
    total_score = np.mean(total_score_list)

    gesture_recall = np.mean(gesture_recall_list)
    force_recall = np.mean(force_recall_list)
    total_recall = np.mean(total_recall_list)

    gesture_f1 = np.mean(gesture_f1_list)
    force_f1 = np.mean(force_f1_list)
    total_f1 = np.mean(total_f1_list)

    if flag ==0:
        return [gesture_score, force_score, total_score], [gesture_recall, force_recall, total_recall],[gesture_f1, force_f1, total_f1]
    if flag ==1:
        return gesture_true_label, gesture_predicted_label, force_true_label, force_predicted_label

