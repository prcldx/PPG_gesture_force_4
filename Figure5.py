""" 对比不同的subject在综合分类中的鲁棒性"""

""" 将总体作为研究对象"""

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
import warnings
warnings.filterwarnings("ignore")

sub_num = 14
# sub_score_array = []
# for sub_index in range(1,sub_num+1):
#     print(sub_index)
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
#
#     # 构造分类器的集合
#     knn = KNeighborsClassifier(n_neighbors=10)  # KNN Modeling
#     clf = svm.SVC(kernel='rbf', C=10)
#     rf = RandomForestClassifier(n_estimators=20)  # RF Modeling
#     lda = LinearDiscriminantAnalysis()
#
#     selected_model = [rf, lda, knn, clf]
#
#     # 循环选择模型
#     output_flag_list = [0]
#     for output_flag in output_flag_list:
#         model_index = 2
#         candidate_model_a = selected_model[model_index]  # 选择只针对手势分类的分类器
#         candidate_model_b = selected_model[model_index]  # 选择只针对力分类的分类器
#         candidate_model_c = selected_model[model_index]  # 选择总共的分类器
#
#         acc_score, recall_score, f1_score = k_fold_multiaim_validate(dataset, label, candidate_model_a, candidate_model_b, candidate_model_c, flag=output_flag)
#         score_list = [acc_score, recall_score, f1_score]
#         model_score_array = np.array(score_list)#二维数组（评估类型×任务）
#     sub_score_array.append(model_score_array)
#     print(sub_index)
#
# sub_score_array = np.array(sub_score_array)#0维：subject 1维度:acc recall f1 2维：3个g , f, all
# sub_score_array = np.squeeze(sub_score_array)
# np.save('./new_figure/sub_score_array_knn.npy', sub_score_array)
# print(1)


RA_flag = 2
if RA_flag == 1:
    #绘制手势分类关于分类器和subject的变化图 F6
    sub_num = 14
    rf_score = np.load('./new_figure/sub_score_array_rf.npy')
    lda_score = np.load('./new_figure/sub_score_array_lda.npy')
    knn_score = np.load('./new_figure/sub_score_array_knn.npy')
    svm_score = np.load('./new_figure/sub_score_array_svm.npy')

    model_score_list = [rf_score, lda_score, knn_score, svm_score]
    gesture_acc_list = []
    for i in range(4):
        gesture_acc_list.append(model_score_list[i][:,0,0])
    gesture_acc_list = np.array(gesture_acc_list)
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
    #color_list = ['#bae7ff', '#91d5ff', '#69c0ff', '#40a9ff']
    color_list = ['#cdb4db','#ffc8dd','#ffafcc','#bde0fe']
    #color_list = ['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    marker_list = ['o', '^', 's','*']
    model_name = ['RF', 'LDA', 'KNN', 'SVM']
    subjects_name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9','S10','S11','S12','S13','S14','Ave']
    ax.set_ylim(0,100)
    ax.set_ylabel('Recognition Accuracy (%)')
    for i in range(4):
        x = np.arange(1, sub_num+1)
        y = gesture_acc_list[i,:]
        ax.plot(x,y*100, color = color_list[i], marker = marker_list[i], label = model_name[i], markersize = 2)
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.1)
    mean_z = np.mean(gesture_acc_list, axis=1)
    std_z = np.std(gesture_acc_list, axis=1, ddof=1)
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
    #plt.savefig('C:\\Users\\86156\\Desktop\\f5_1.png')
    plt.savefig('C:\\Users\\Lenovo\\Desktop\\fig\\f5_2.png')


if RA_flag == 2:
    #绘制力分类精度关于分类器和subject的图f51
    sub_num = 14
    rf_score = np.load('./new_figure/sub_score_array_rf.npy')
    lda_score = np.load('./new_figure/sub_score_array_lda.npy')
    knn_score = np.load('./new_figure/sub_score_array_knn.npy')
    svm_score = np.load('./new_figure/sub_score_array_svm.npy')

    model_score_list = [rf_score, lda_score, knn_score, svm_score]
    gesture_acc_list = []
    for i in range(4):
        gesture_acc_list.append(model_score_list[i][:,0,1])
    gesture_acc_list = np.array(gesture_acc_list)
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
    #color_list = ['#bae7ff', '#91d5ff', '#69c0ff', '#40a9ff']
    #color_list = ['#e6f7ff', '#bae7ff', '#91d5ff', '#69c0ff']
    #color_list = ['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    color_list = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe']
    marker_list = ['o', '^', 's','*']
    model_name = ['RF', 'LDA', 'KNN', 'SVM']
    subjects_name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9','S10','S11','S12','S13','S14','Ave']
    ax.set_ylim(0,100)
    ax.set_ylabel('Recognition Accuracy (%)')
    for i in range(4):
        x = np.arange(1, sub_num+1)
        y = gesture_acc_list[i,:]
        ax.plot(x,y*100, color = color_list[i], marker = marker_list[i], label = model_name[i], markersize = 2)
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.1)
    mean_z = np.mean(gesture_acc_list, axis=1)
    std_z = np.std(gesture_acc_list, axis=1, ddof=1)
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
    plt.savefig('C:\\Users\\86156\\Desktop\\f5_2.png')#关于力分类在所有人上的图
    #plt.savefig('C:\\Users\\Lenovo\\Desktop\\fig\\f5_2.png')




# sub_num = 9
# sub_score_array = np.load('./new_figure/sub_score_array.npy')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams.update({'font.size': 6})
# fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (3.5, 4.5), dpi = 400)
# x = np.arange(10)
# y = np.arange(10)
# color_list = ['skyblue', 'lightcoral', 'navajowhite']
# task_name = ['Gesture Recognition', 'Force Level Recognition', 'Gesture and Force level Recognition ']
# marker_list = ['o','^','s']
# subjects_name = ['S1', 'S2','S3', 'S4','S5', 'S6','S7','S8', 'S9', 'Ave']
# label_name = ['Precision (%)','Recall (%)','F1-score (%)' ]
# for i in range(3):
#     x = np.arange(1,sub_num+1)
#     y = sub_score_array[:, i, :]
#     ax[i].set_ylim(0, 100)
#     ax[i].set_ylabel(label_name[i])
#     for j in range(3):
#         ax[i].plot(x,y[:,j]*100,color = color_list[j], marker = marker_list[j], label = task_name[j], markersize = 2)
#     if i==2:
#         ax[i].legend()
#     mean_z = np.mean(y,axis=0)
#     std_z = np.std(y, axis = 0, ddof=1)
#     for m in range(3):
#         ax[i].bar([10+0.5*m], mean_z[m] * 100, width=0.4, lw=3, color=color_list[m])
#         ax[i].errorbar([10+0.5*m], mean_z[m] * 100,  yerr=std_z[m] * 100, fmt='.', ecolor='black',
#                   elinewidth=0.3, ms=0.1, mfc='wheat', mec='salmon', capsize=2.5, capthick=0.4)
#     tic = list(x)
#     tic.append(10.5)
#     ax[i].set_xticks(tic, subjects_name)
#     #ax[i].set_xticklabels(tic, subjects_name)
#     #ax[i].set_sticklabels(tic, subjects_name)
# plt.tight_layout()
# plt.savefig('C:\\Users\\86156\\Desktop\\f5.png')
# plt.show()


#
#
#
# RA_flag =0
# if RA_flag==1:
#     labels_name = ['Sub 1', 'Sub 2','Sub 3', 'Sub 4','Sub 5', 'Sub 6']
#     task_name = ['Gesture Recognition', 'Force Level Recognition', 'Gesture and Force level Recognition ']
#     color_list = ['skyblue', 'lightcoral', 'navajowhite']
#     for i in range(3):
#         x = np.arange(sub_num)
#         y = sub_score_array[:, i]
#         plt.bar(x + 0.2 * i, y * 100, alpha=0.6, width=0.2, lw=3, label=task_name[i], color=color_list[i])
#
#     plt.xticks(x + 0.2, labels_name, fontsize=12)
#     plt.xlabel('Subject', loc='center', fontsize=12, weight='medium')
#     plt.ylabel('Recall (%) ', fontsize=12)
#     plt.legend(loc = 'lower right')
#     plt.ylim(0, 100)
#     plt.show()





