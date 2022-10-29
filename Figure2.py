"""对每个手势中的力进行分类 ,画出不同的分类器相对于力分类准确率的对比，最后得到一个表格"""


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
from Tool_Functionset import *

knn = KNeighborsClassifier()  # KNN Modeling
clf = svm.SVC(kernel='rbf',C = 10)
rf = RandomForestClassifier(n_estimators=20)  # RF Modeling
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]

#绘图选项
CM_flag = 1; #0表示不绘制混淆矩阵，1表示绘制混淆矩阵
RA_flag = 0; #0表示不会绘制，1表示会绘制

model_num = 4
sub_num = 14
gesture_num = 4

# 出在各个手势下进行力分类的表格
# ACC_score_list = np.zeros((sub_num*4,gesture_num))# 4代表的是有4种信号类型
# for model_index in range(model_num):
#     # 选择对应的模型
#     feature_set = {'PPG': range(0, 15), 'PPG_R': range(0, 5), 'PPG_IR': range(5, 10), 'PPG_G': range(10, 15)}
#     feature_set_cata = ['PPG', 'PPG_R', 'PPG_IR', 'PPG_G']
#
#     #选择用哪些通道的特征进行训练
#     ACC_overfeature_list = np.zeros((sub_num,gesture_num))
#     for feature_index in range(4):
#         feature = feature_set[feature_set_cata[feature_index]]
#
#         # 从头开始遍历所有的被试
#         ACC_oversub_list = []
#         for sub_index in range(1, sub_num + 1):
#             path1 = './Feature_2/data' + str(sub_index) + '3.csv'
#             path2 = './Feature_2/label' + str(sub_index) + '3.csv'
#             #
#             dataset = pd.read_csv(path1, header=None)
#             label = pd.read_csv(path2, header=None)
#             dataset = np.array(dataset)
#             label = np.array(label)
#             label = label.astype(int)
#             #
#             dataset = dataset[:, feature]
#             # 打乱数据，为各个训练器准备训练数据
#             data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
#             data_pool[:, 0:dataset.shape[1]] = dataset
#             data_pool[:, dataset.shape[1]:] = label
#             np.random.seed(0)
#             np.random.shuffle(data_pool)
#             dataset = data_pool[:, 0:dataset.shape[1]]
#             label = data_pool[:, dataset.shape[1]:]
#             label = label.astype(int)
#
#             """ 构造分类器的集合"""
#
#             candidate_model_a = selected_model[model_index]
#             candidate_model_b = selected_model[model_index]
#             force_score_kfold_array, gesture_score_kfold_array = k_fold_validate(dataset, label, candidate_model_a,
#                                                                                  candidate_model_b)
#
#             ACC_oversub_list.append(list(force_score_kfold_array))# sub_num*gesture_num
#
#         ACC_oversub_list = np.array(ACC_oversub_list)
#         if feature_index ==0:
#             ACC_overfeature_list = ACC_oversub_list
#         else:
#             ACC_overfeature_list = np.vstack((ACC_overfeature_list, ACC_oversub_list))
#
#     print(model_index)
#
#     if model_index ==0:
#         ACC_score_list = ACC_overfeature_list
#     else:
#         ACC_score_list = np.vstack((ACC_score_list, ACC_overfeature_list))
#
# print(ACC_score_list.shape)
# np.save('./new_figure/Force_score_list', ACC_score_list)


#从上述文件中读取数据画图


# ACC_score_list = np.zeros((sub_num*4,gesture_num))# 4代表的是有4种信号类型
# for model_index in range(3,4):
#     # 选择对应的模型
#     feature_set = {'PPG': range(0, 15), 'PPG_R': range(0, 5), 'PPG_IR': range(5, 10), 'PPG_G': range(10, 15)}
#     feature_set_cata = ['PPG', 'PPG_R', 'PPG_IR', 'PPG_G']
#
#     #选择用哪些通道的特征进行训练
#     ACC_overfeature_list = []
#     for feature_index in range(4):
#         feature = feature_set[feature_set_cata[feature_index]]
#
#         # 从头开始遍历所有的被试
#         ACC_oversub_list = []
#         for sub_index in range(1, sub_num + 1):
#             path1 = './Feature_2/data' + str(sub_index) + '3.csv'
#             path2 = './Feature_2/label' + str(sub_index) + '3.csv'
#             #
#             dataset = pd.read_csv(path1, header=None)
#             label = pd.read_csv(path2, header=None)
#             dataset = np.array(dataset)
#             label = np.array(label)
#             label = label.astype(int)
#             #
#             dataset = dataset[:, feature]
#             # 打乱数据，为各个训练器准备训练数据
#             data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
#             data_pool[:, 0:dataset.shape[1]] = dataset
#             data_pool[:, dataset.shape[1]:] = label
#             np.random.seed(0)
#             np.random.shuffle(data_pool)
#             dataset = data_pool[:, 0:dataset.shape[1]]
#             label = data_pool[:, dataset.shape[1]:]
#             label = label.astype(int)
#
#             """ 构造分类器的集合"""
#
#             candidate_model_a = selected_model[model_index]  # 选择只针对手势分类的分类器
#             candidate_model_b = selected_model[model_index]  # 选择只针对力分类的分类器
#             candidate_model_c = selected_model[model_index]  # 选择总共的分类器
#             #
#             output_flag = 0
#             acc_score, recall_score, f1_score = k_fold_multiaim_validate(dataset, label, candidate_model_a,
#                                                                          candidate_model_b, candidate_model_c,
#                                                                          flag=output_flag)
#             ACC_oversub_list.append(acc_score[1])
#
#         print(sub_index)
#         ACC_overfeature_list.append(ACC_oversub_list)
#
#     print(feature_index)
#     ACC_overfeature_list = np.array(ACC_overfeature_list)
#
# print(ACC_overfeature_list.shape)
# np.save('./new_figure/ACC_overfeature_list', ACC_overfeature_list)

RA_flag=1
if RA_flag == 1:
    #绘制力分类精度关于不同和subject的图
    sub_num = 14
    ACC_overfeature_list = np.load('./new_figure/ACC_overfeature_list.npy')
    print(1)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(3.5, 2), dpi=800)
    ax = plt.gca()
    bwith = 0.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    color_list = ['#bae7ff', '#91d5ff', '#69c0ff', '#40a9ff']
    marker_list = ['o', '^', 's','*']
    model_name = ['PPG-ALL', 'PPG-R', 'PPG-IR', 'PPG-G']
    subjects_name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9','S10','S11','S12','S13','S14','Ave']
    ax.set_ylim(0,100)
    ax.set_ylabel('Recognition Accuracy (%)')
    for i in range(4):
        x = np.arange(1, sub_num+1)
        y = ACC_overfeature_list[i,:]
        ax.plot(x,y*100, color = color_list[i], marker = marker_list[i], label = model_name[i], markersize = 2)
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.1)
    mean_z = np.mean(ACC_overfeature_list, axis=1)
    std_z = np.std(ACC_overfeature_list, axis=1, ddof=1)
    for m in range(4):
        ax.bar([15 + 0.5 * m], mean_z[m] * 100, width=0.4, lw=3, color=color_list[m])
        ax.errorbar(15 + 0.5 * m, mean_z[m] * 100, yerr=[[0], [std_z[m] * 100]], fmt='.', ecolor='black',
                       elinewidth=0.3, ms=0.001, mfc='wheat', mec='salmon', capsize=1, capthick=0.2)
        ax.scatter([15 + 0.5 * m], 5, marker=marker_list[m], s=4, c = '#FFFFFF')

    x = np.arange(1,sub_num+1)
    tic = list(x)
    tic.append(15.75)
    ax.set_xticks(tic, subjects_name)
    plt.tight_layout()
    #plt.savefig('C:\\Users\\86156\\Desktop\\f2_1.png')#关于力分类在所有人上的图
    plt.savefig('C:\\Users\\Lenovo\\Desktop\\fig\\f5_3.png')


#
# ACC_score_list = np.load('./new_figure/Force_score_list.npy')
#
# Force_score_list = []
# mean_force_arr = np.zeros((16,4))
# std_force_arr = np.zeros((16,4))
#
# for r in range(int(ACC_score_list.shape[0]/sub_num)):
#     for c in range(4):
#         data = ACC_score_list[range(r*sub_num, r*sub_num+sub_num),c]
#         mean_force = np.mean(data)
#         std_force = np.std(data,ddof=1)
#
#         mean_force_arr[r,c] = mean_force
#         std_force_arr[r,c] = std_force
#         # = [str(np.mean(data)) + '±' + str(np.std(data,ddof=1))]
#         #Force_score_list.append(mean_std)
#
# print(1)
# np.savetxt('./new_figure/force_1.csv', mean_force_arr*100, delimiter=',', fmt='%.5e')
# np.savetxt('./new_figure/force_2.csv', std_force_arr*100, delimiter=',', fmt='%.5e')



