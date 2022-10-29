""" 比率变换对于识别精度的变化, 已经改成了lot的交叉验证的形式"""

import numpy as np
from sklearn.decomposition import PCA
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

from Tool_Functionset import *


def multi_set_size(dataset, label, candidate_model_a, candidate_model_b, candidate_model_C, flag, frac):
    """
    目的是: 为model_a，和model_b选择不同的分类器，验证在lot验证下的分类精度
    :param dataset:
    :param label:
    :return: force_score_lot_array; gesture_score_lot_array; total_score_lot_array

    """
    gesture_score_list = []  # 记录每一次交叉验证的手势分类精度
    force_score_list = [] #记录每一次力分类的精度
    total_score_list = [] #记录总共的分类精度

    gesture_recall_list = []
    force_recall_list = [] #记录每一次力分类的精度
    total_recall_list = [] #记录总共的分类精度

    fold = 10
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
        batch_size = int(np.floor(frac * train_dataset.shape[0]))

        train_dataset = train_dataset[0:batch_size,:]
        train_label = train_label[0:batch_size,:]

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
        test_recall_a = sm.recall_score(y_true=test_label_a, y_pred=predict_a, average='macro')

        gesture_score_list.append(test_score_a)  # 把lot交叉验证的分数记录下来
        gesture_recall_list.append(test_recall_a)

        """ 力分类"""
        # 设置model b 的标签
        train_label_b = train_label[:, 2]
        test_label_b = test_label[:, 2]
        model_b.fit(train_dataset, train_label_b)

        #
        predict_b = model_b.predict(test_dataset)  # 给出第一次预测的结果
        test_score_b = model_b.score(test_dataset, test_label_b)
        force_score_list.append(test_score_b)  # 把lot交叉验证的分数记录下来

        test_recall_b = sm.recall_score(y_true=test_label_b, y_pred=predict_b, average='macro')
        force_recall_list.append(test_recall_b)

        """ 总分类 """
        #设置model_c的标签
        train_label_c = train_label[:, 4]
        test_label_c = test_label[:, 4]
        model_c.fit(train_dataset, train_label_c)

        #
        predict_c = model_c.predict(test_dataset)  # 给出第一次预测的结果
        test_score_c = model_c.score(test_dataset, test_label_c)
        total_score_list.append(test_score_c)  # 把lot交叉验证的分数记录下来

        test_recall_c = sm.recall_score(y_true=test_label_c, y_pred=predict_c, average='macro')
        total_recall_list.append(test_recall_c)


    gesture_score = np.mean(gesture_score_list)
    force_score = np.mean(force_score_list)
    total_score = np.mean(total_score_list)

    gesture_recall = np.mean(gesture_recall_list)
    force_recall = np.mean(force_recall_list)
    total_recall = np.mean(total_recall_list)


    if flag ==1:
        return gesture_score, force_score, total_score
    else:
        return gesture_recall, force_recall, total_recall


sub_num = 14
# rate_score_array = []
# for sub_index in range(1,sub_num+1):
#     path1 = './Feature_2/data' + str(sub_index) + '3.csv'
#     path2 = './Feature_2/label' + str(sub_index) + '3.csv'
#
#     dataset = pd.read_csv(path1, header=None)
#     label = pd.read_csv(path2, header=None)
#     dataset = np.array(dataset)
#     label = np.array(label)
#     label = label.astype(int)
#
#     # 打乱数据，为各个训练器准备训练数据
#     data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
#     data_pool[:, 0:dataset.shape[1]] = dataset
#     data_pool[:, dataset.shape[1]:] = label
#     np.random.seed(0)  # 控制随机数产生的种子
#     np.random.shuffle(data_pool)
#     dataset = data_pool[:, 0:dataset.shape[1]]
#     label = data_pool[:, dataset.shape[1]:]
#     label = label.astype(int)
#
#     # 构造分类器的集合
#     knn = KNeighborsClassifier()  # KNN Modeling
#     clf = svm.SVC(kernel='rbf', C=10)
#     rf = RandomForestClassifier(n_estimators=16)  # RF Modeling
#     lda = LinearDiscriminantAnalysis()
#     selected_model = [rf, lda, knn, clf]
#     force_score_classifier_array = np.zeros((4, 10))
#     gesture_score_classifier_array = np.zeros((4, 10))
#
#     # 循环选择模型
#     model_score_array = []
#     output_flag = 1
#     frac_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#     for frac in frac_list:
#         #根据预实验的结果，分类器选择SVM分类器
#         candidate_model_a = selected_model[3]  # 选择只针对手势分类的分类器
#         candidate_model_b = selected_model[3]  # 选择只针对力分类的分类器
#         candidate_model_c = selected_model[3]  # 选择总共的分类器
#
#
#         gesture_score, force_score, total_score = multi_set_size(dataset, label, candidate_model_a, candidate_model_b,
#                                                                  candidate_model_c, flag=output_flag, frac=frac)
#         print(frac)
#         score_list = [gesture_score, force_score, total_score]
#         model_score_array.append(score_list)
#     model_score_array = np.array(model_score_array)
#     rate_score_array.append(model_score_array)
#     print(sub_index)
#
# rate_score_array = np.array(rate_score_array)#0维：subject 1维度:frac 2维：3个score
# np.save('./new_figure/rate_score_array.npy', rate_score_array)
# print(1)

#
rate_score_array = np.load('./new_figure/rate_score_array.npy')


linestyle_list = ['-','--','-.']
marker_list = ['o','*','1']
color_list = ['skyblue', 'lightcoral', 'navajowhite']
legend_list = ['Hand gesture classification', 'Force level classification', 'Simultaneous classification of hand gestures and force levels ']
xlabel = ['10%', '20%', '30%','40%','50%','60%','70%','80%','90%','100%']
RA_flag =1
if RA_flag ==1:
    data = np.mean(rate_score_array, axis=0)
    data_max = np.max(rate_score_array, axis=0)
    data_min = np.min(rate_score_array, axis=0)
    data_std = np.std(rate_score_array, axis=0)
    print(1)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(3.5, 3), dpi=800)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    bwith = 0.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    x = np.arange(data.shape[0])
    for i in range(3):
        plt.scatter(x, data[:, i] * 100, marker=marker_list[i], s=5)
        plt.plot(x, data[:,i]*100,label = legend_list[i], color= color_list[i], linestyle=linestyle_list[i] )
        # plt.fill_between(x, (data[:,i]+data_std[:,i])*100, (data[:,i]-data_std[:,i])*100,  # 上限，下限
        #                  facecolor=color_list[i],  # 填充颜色
        #                  alpha=0.3)  # 透明度

        plt.ylim([60,90])
        plt.xticks(x,xlabel,fontsize = 6)
        plt.tick_params(width=0.5)  # 设置刻度线条的粗细
        plt.grid(axis='y')

    plt.ylabel('Recognition Accuracy(%)', fontsize = 7)
    plt.xlabel('Percentage of Training set', fontsize = 7)
    #plt.legend(loc =  'lower right',fontsize = 6)
    leg = plt.legend(loc = 'upper left',bbox_to_anchor = (-0.01,-0.25) ,fontsize = 7, ncol = 1)
    leg.get_frame().set_linewidth(0.1)
    plt.tight_layout()
    #plt.savefig('C:\\Users\\86156\\Desktop\\f6_1.png')
    plt.savefig('C:\\Users\\86156\\Desktop\\why\\f6_new_2.pdf',bbox_inches='tight', pad_inches=0.1)
    #plt.savefig('C:\\Users\\Lenovo\\Desktop\\fig\\f6_new_2.png',bbox_inches='tight', pad_inches=0.1)#,bbox_inches='tight'

