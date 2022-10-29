""" 比较在力变化时，手势精度的分类情况,
 单独将不同等级的力作为独立的输入，监测在不同独立力等级输入下手势分类"""

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

# 数据处理程序
# sub_num = 9
# gesture_oversub_score = []
#
# for sub_index in range(1,sub_num+1):
#
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
#     rf = RandomForestClassifier(n_estimators=20)  # RF Modeling
#     lda = LinearDiscriminantAnalysis()
#     selected_model = [rf, lda, knn, clf]
#
#     gesture_score_classifier_array = np.zeros((4, 10))# ml_model * fold
#
#     # 探究不同力变化时，对于手势分类精度的影响
#     gesture_overforce_score = []
#     for force_index in range(4):
#         if force_index == 3:
#             new_dataset = dataset
#             new_label = label
#         else:
#             new_dataset = dataset[label[:, 2] == force_index]
#             new_label = label[label[:, 2] == force_index]
#
#         # 循环选择模型
#         for model_index in range(4):
#             candidate_model_a = selected_model[model_index]
#             gesture_score_lot_array = gesture_lot_validate(new_dataset, new_label, candidate_model_a)
#             gesture_score_classifier_array[model_index, :] = gesture_score_lot_array
#
#         gesture_mean_score = np.mean(gesture_score_classifier_array, axis=1)
#         gesture_overforce_score.append(gesture_mean_score)
#         print('手势分类的平均准确率\n', np.mean(gesture_score_classifier_array, axis=1))
#         print(force_index)
#
#     gesture_overforce_score = np.array(gesture_overforce_score)
#     print(sub_index)
#     gesture_oversub_score.append(gesture_overforce_score)
#
# gesture_oversub_score = np.array(gesture_oversub_score)#dim0= sub, dim1 = force type , dim2 = model type
#
# np.save('./new_figure/gesture_oversub_score.npy', gesture_oversub_score)
#
# print(1)



#画图程序
gesture_oversub_score = np.load('./new_figure/gesture_oversub_score.npy')
RA_flag = 3
#绘制柱状图
if RA_flag==1:
    MVC_10 = gesture_oversub_score[:,0,:]
    MVC_40 = gesture_oversub_score[:,1,:]
    MVC_70 = gesture_oversub_score[:,2,:]
    MVC_all = gesture_oversub_score[:,3,:]

    print(1)
    feature_set_cata = [MVC_10, MVC_40, MVC_70, MVC_all]
    labels_name = ['RF', 'LDA', 'KNN', 'SVM']
    MVC_level = ['MVC_10', 'MVC_40', 'MVC_70', 'MVC_all']

    color_list = ['skyblue', 'lightcoral', 'navajowhite', 'limegreen']

    for i in range(4):
        x = np.arange(4)
        y = np.mean(feature_set_cata[i], axis=0)
        error = np.std(feature_set_cata[i], axis=0)
        plt.bar(x + 0.2 * i, y * 100, alpha=0.6, width=0.2, lw=3, label=MVC_level[i], color=color_list[i])
        plt.errorbar(x + 0.2 * i, y * 100, yerr=error * 100, fmt='.', ecolor='black',
                     elinewidth=1, ms=5, mfc='wheat', mec='salmon', capsize=5)
    plt.xticks(x + 0.3, labels_name, fontsize=12)
    plt.xlabel('Machine learning Model', loc='center', fontsize=12, weight='medium')
    plt.ylabel('Recognition Accuracy (%) ', fontsize=12)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('C:\\Users\\86156\\Desktop\\f31.png')
    plt.show()

#改进柱状图
RA_flag =3
if RA_flag==3:
    MVC_10 = gesture_oversub_score[:, 0, :]
    MVC_40 = gesture_oversub_score[:, 1, :]
    MVC_70 = gesture_oversub_score[:, 2, :]
    MVC_all = gesture_oversub_score[:, 3, :]
    print(1)

    feature_set_cata = [MVC_10, MVC_40, MVC_70, MVC_all]
    labels_name = ['RF', 'LDA', 'KNN', 'SVM']
    MVC_level = ['MVC_10', 'MVC_40', 'MVC_70', 'MVC_all']

    color_list = ['skyblue', '#ffadad', '#ffd6a5', '#caffbf']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(3.5,3), dpi=600)

    ax = plt.gca()
    bwith = 0.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    #, alpha=0.6
    for i in range(4):
        x = np.arange(4)*1.2
        y = np.mean(feature_set_cata[i], axis=0)
        error = np.std(feature_set_cata[i], axis=0, ddof=1)
        plt.bar(x + 0.2 * i, y * 100, width=0.18, lw=3, label=MVC_level[i], color=color_list[i])
        plt.errorbar(x + 0.2 * i, y * 100, yerr=error * 100, fmt='.', ecolor='black',
                     elinewidth=0.3, ms=0.1, mfc='wheat', mec='salmon', capsize=1, capthick = 0.2)
    plt.xticks(x + 0.3, labels_name, fontsize=6)
    plt.xlabel('Machine learning Model', loc='center', fontsize=6, weight='medium')
    plt.ylabel('Recognition Accuracy (%) ', fontsize=6)
    plt.tick_params(width=0.5)#设置刻度线条的粗细
    plt.legend()
    #plt.legend(loc = 'upper left',bbox_to_anchor = (-0.02,-0.15) ,fontsize = 6, ncol = 4)
    plt.ylim(0, 102)
    plt.tight_layout()
    plt.savefig('C:\\Users\\86156\\Desktop\\f32.png')
    #plt.savefig('C:/Users/Lenovo/Desktop/New_Figure')
    plt.show()




#绘制箱线图

RA_flag = 1
if RA_flag==2:
    RF = gesture_oversub_score[:, :, 0]
    LDA = gesture_oversub_score[:, :, 1]
    KNN = gesture_oversub_score[:, :, 2]
    SVM = gesture_oversub_score[:, :, 3]
    print(1)

    data_list = [RF, LDA, KNN, SVM]
    # 首先有图（fig），然后有轴（ax）
    posi = np.arange(4)

    title_list = ['RF', 'LDA', 'KNN', 'SVM']
    colors_list = ['pink', 'lightblue', 'lightgreen','lightcoral']
    label_list = ['MVC_10','MVC_40','MVC_70','MVC_all']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(3.5,3), dpi=200)
    for r in range(2):
        for c in range(2):
            data_index=r*2+c
            data = data_list[data_index]
            bplot1 = ax[r, c].boxplot(data*100, vert=True, showfliers=False,
                                      patch_artist=True)
            ax[r, c].set_xticks(posi+1, label_list, rotation = -30)
            #plt.xticks(x + 0.6, labels_name, fontsize=6)
            ax[r, c].set_title(title_list[data_index])
            ax[r,c].set_ylim([50,105])
            ax[r,c].set_yticks([50,60,70,80,90,100])
            if c==0:
                ax[r, c].set_ylabel('Recognition Accuracy(%)')
            #plt.ylim([50,100])
            # ax.set_label("a")
            for i in range(len(colors_list)):
                bplot1['boxes'][i].set_facecolor(colors_list[i])
    plt.tight_layout()
    plt.savefig('C:\\Users\\86156\\Desktop\\f3.png')
    plt.show()







