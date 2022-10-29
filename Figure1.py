""" 使用一般的机器学习模型进行 "综合分类" 的评估，出混淆矩阵的图,K-fold validation"""

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

""" 构造分类器的集合(这是经过网格调优之后的)"""

knn = KNeighborsClassifier(n_neighbors =  10)                #KNN Modeling
clf = svm.SVC(kernel='rbf', C=10)
rf = RandomForestClassifier(n_estimators=20)       #RF Modeling
rf = RandomForestClassifier()
lda = LinearDiscriminantAnalysis()
#selected_model = [rf, lda, knn, clf]
selected_model = [clf]

#绘图选项
CM_flag = 1 #0表示不绘制混淆矩阵，1表示绘制混淆矩阵
RA_flag = 1 #0表示不会绘制，1表示会绘制
#
model_num = 1
sub_num = 14
feature_num = 1 #表示选择了几种信号

# ACC_score_list = np.zeros((model_num,feature_num*sub_num))
# for model_index in range(model_num):
#     # 选择对应的模型
#     #feature_set = {'PPG': range(0, 18), 'PPG_R': range(0, 6), 'PPG_IR': range(6, 12), 'PPG_G': range(12, 18)}
#     feature_set = {'PPG': range(0, 15), 'PPG_R': range(0, 5), 'PPG_IR': range(5, 10), 'PPG_G': range(10, 15),"PPG_R+IR": range(0,10), "PPG_R+G": [0,1,2,3,4,10,11,12,13,14],"PPG_IR+G": range(5,15)}
#     feature_set_cata = ['PPG', 'PPG_R', 'PPG_IR', 'PPG_G',"PPG_R+IR", "PPG_R+G","PPG_IR+G"]
#
#     #选择用哪些通道的特征进行训练
#     for feature_index in range(feature_num):
#         feature = feature_set[feature_set_cata[feature_index]]
#
#         # 从头开始遍历所有的被试
#         test_label_oversub_list = []
#         predicted_label_oversub_list = []
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
#             fold = 10  # 总共进行5次验证结果
#             #kf = KFold(n_splits=fold)
#             predicted_label_overkfold_list = []
#             test_label_overkfold_list = []
#             for i in range(fold):
#                 j = i
#                 test_index = label[:, 3] == i
#                 test_index_1 = label[:, 3] == j
#                 for m in range(test_index_1.shape[0]):
#                     test_index[m] = test_index[m] or test_index_1[m]
#                 train_index = ~ test_index
#             #for train_index, test_index in kf.split(dataset):
#                 train_dataset = dataset[train_index, :]
#                 train_label = label[train_index, :]
#                 train_label = train_label[:, 4]
#                 test_dataset = dataset[test_index, :]
#                 test_label = label[test_index, :]
#                 test_label = test_label[:, 4]
#                 # 选择模型，实际上，在绘制FIG_1时，只需要选择一个模型就可以
#                 model = selected_model[model_index]
#
#                 # 进行归一化
#                 scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_dataset)
#                 train_dataset = scaler.transform(train_dataset)
#                 test_dataset = scaler.transform(test_dataset)
#                 model.fit(train_dataset, train_label)
#                 predicted_label = model.predict(test_dataset)
#                 test_label_overkfold_list.append(list(test_label))
#                 predicted_label_overkfold_list.append(list(predicted_label))
#
#             test_label_overkfold_list = np.array(test_label_overkfold_list)
#             predicted_label_overkfold_list = np.array(predicted_label_overkfold_list)
#             test_label_overkfold_list = np.ravel(test_label_overkfold_list)
#             predicted_label_overkfold_list = np.ravel(predicted_label_overkfold_list)
#             test_label_oversub_list.append(list(test_label_overkfold_list))
#             predicted_label_oversub_list.append(list(predicted_label_overkfold_list))
#
#             scores = sm.accuracy_score(test_label_overkfold_list, predicted_label_overkfold_list)#把所有的交叉验证平均值存为accuracy score
#             ACC_score_list[model_index, feature_index * sub_num +(sub_index-1)] = scores
#
#         test_label_oversub_list = np.array(test_label_oversub_list)
#         test_label_oversub_list = np.ravel(test_label_oversub_list)
#
#         predicted_label_oversub_list = np.array(predicted_label_oversub_list)
#         predicted_label_oversub_list = np.ravel(predicted_label_oversub_list)
#
#         np.save('./new_figure/test_label_oversub_list.npy', test_label_oversub_list)
#         np.save('./new_figure/predicted_label_oversub_list.npy', predicted_label_oversub_list)
#
#
#
#         if (CM_flag ==1) and (model_index==0):
#             m = sm.confusion_matrix(test_label_oversub_list, predicted_label_oversub_list)
#             print(m)
#             label_strlist = ['TIP10', 'TIP40', 'TIP70', 'TMP10', 'TMP40', 'TMP70', 'TRP10',
#                              'TRP40',
#                              'TRP70', 'KP10', 'KP40', 'KP70' ]
#             plot_confusion_matrix(m, label_strlist,
#                                   'Confusion Matrix for hand Gestures at 3 Force levels' + ' (' + feature_set_cata[
#                                       feature_index] + ') ')
#             plt.show()
#             plt.savefig('C:\\Users\\86156\\Desktop\\fc1.png')
#             print(1)
#     print(model_index)
#
# np.save('./new_figure/figure_1.npy', ACC_score_list)



# print(ACC_score_list)
# np.savetxt('./figure_note/ACC_score_list', ACC_score_list, delimiter=',')
#选择是否绘制识别准确率的图片


# test_label_oversub_list = np.load('./new_figure/test_label_oversub_list.npy')
# predicted_label_oversub_list = np.load('./new_figure/predicted_label_oversub_list.npy')
#
# m = sm.confusion_matrix(test_label_oversub_list, predicted_label_oversub_list)
# print(m)
# label_strlist = ['TIP10', 'TIP40', 'TIP70', 'TMP10', 'TMP40', 'TMP70', 'TRP10',
#                         'TRP40',
#                         'TRP70', 'KP10', 'KP40', 'KP70' ]
# plot_confusion_matrix(m, label_strlist,
#                                   'Confusion Matrix for hand Gestures at 3 Force levels')
# plt.savefig('C:\\Users\\86156\\Desktop\\fc1_1.png')



#直接设置双框3.5英寸的标准大小,绘制多种模式PPG对比的图片
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 8})

ACC_score_list = np.load('./new_figure/figure_1.npy')
feature_num = 7
RA_flag =1
if RA_flag==1:
    PPG = ACC_score_list[:, range(sub_num)]
    PPG_R = ACC_score_list[:, range(sub_num, 2*sub_num)]
    PPG_IR = ACC_score_list[:, range(2*sub_num, 3*sub_num)]
    PPG_G = ACC_score_list[:, range(3*sub_num, 4*sub_num)]
    PPG_RIR = ACC_score_list[:, range(4*sub_num, 5*sub_num)]
    PPG_RG = ACC_score_list[:, range(5*sub_num, 6*sub_num)]
    PPG_IRG = ACC_score_list[:, range(6*sub_num, 7*sub_num)]
    print(1)
    feature_set_cata = [PPG, PPG_R, PPG_IR, PPG_G, PPG_RIR, PPG_RG, PPG_IRG]
    labels_name = ['RF', 'LDA', 'KNN', 'SVM']
    sensor_name = ['PPG-ALL', 'PPG-R', 'PPG-IR', 'PPG-G', ' PPG-R+IR','PPG-R+G', 'PPG-IR+G' ]
    #color_list = ['skyblue', 'lightcoral', 'navajowhite', 'limegreen','peru','royalblue','#ffc6ff']ffadad
    #color_list = ['skyblue', '#ffadad', '#ffd6a5', '#caffbf', '#fdffb6', '#a0c4ff', '#ffc6ff']
    #color_list = ['geekblue', '#44c489', '#28a9ae', '#28a2b7', '#4c7788', '#6c4f63', '#432c39']
    color_list = ['#e6f7ff', '#bae7ff', '#91d5ff', '#69c0ff', '#40a9ff', '#1890ff', '#096dd9']
    plt.figure(figsize=(3.5,3.5), dpi=800)
    ax = plt.gca()
    bwith = 0.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    #, alpha=0.6
    for i in range(feature_num):
        x = np.arange(4)*1.6
        y = np.mean(feature_set_cata[i], axis=1)
        error = np.std(feature_set_cata[i], axis=1, ddof=1)
        y_errormin = [0,0,0,0]
        plt.bar(x + 0.2 * i, y * 100, width=0.18, lw=3, label=sensor_name[i], color=color_list[i])
        plt.errorbar(x + 0.2 * i, y * 100, yerr=[y_errormin, error * 100], fmt='.', ecolor='black',
                     elinewidth=0.3, ms=0.000001, mfc='wheat', mec='salmon', capsize=1, capthick=0.2)
    plt.xticks(x + 0.6, labels_name, fontsize=8)
    plt.xlabel('Machine learning Model', loc='center', fontsize=8, weight='medium')
    plt.ylabel('Recognition Accuracy (%) ', fontsize=8)
    plt.tick_params(width=0.5)#设置刻度线条的粗细
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[4], handles[1],handles[5], handles[2], handles[6],handles[3]]
    labels = [labels[0], labels[4], labels[1],labels[5], labels[2], labels[6],labels[3]]
    leg = ax.legend(handles, labels, loc = 'upper left',bbox_to_anchor = (-0.05,-0.15) ,fontsize = 6, ncol = 4)
    leg.get_frame().set_linewidth(0.1)

    plt.ylim(0, 90)
    plt.yticks([0,10,20,30,40,50,60,70,80,90])
    plt.tight_layout()
    plt.savefig('C:\\Users\\86156\\Desktop\\f1_1.png')
    #plt.savefig('C:/Users/Lenovo/Desktop/New_Figure')
    plt.show()


