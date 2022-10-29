""" 手势识别和力识别的混淆矩阵图"""

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
gesture_true_label_list = []
gesture_predicted_label_list = []
force_true_label_list = []
force_predicted_label_list = []

# all_label = []
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
#     output_flag_list = [1]
#     for output_flag in output_flag_list:
#         model_index = 3
#         candidate_model_a = selected_model[model_index]  # 选择只针对手势分类的分类器
#         candidate_model_b = selected_model[model_index]  # 选择只针对力分类的分类器
#         candidate_model_c = selected_model[model_index]  # 选择总共的分类器
#
#         gesture_true_label, gesture_predicted_label, force_true_label, force_predicted_label = k_fold_multiaim_validate(dataset, label, candidate_model_a, candidate_model_b, candidate_model_c, flag=output_flag)
#
#         gesture_true_label_list.extend(gesture_true_label)
#         gesture_predicted_label_list.extend(gesture_predicted_label)
#         force_true_label_list.extend(force_true_label)
#         force_predicted_label_list.extend(force_predicted_label)
#
# all_label.append(gesture_true_label_list)
# all_label.append(gesture_predicted_label_list)
# all_label.append(force_true_label_list)
# all_label.append(force_predicted_label_list)
# all_label = np.array(all_label)
# np.save('./new_figure/two_confusionmatrix.npy', all_label)
# print(1)


all_label = np.load('./new_figure/two_confusionmatrix.npy')
gt = all_label[0,:]
gp = all_label[1,:]
ft = all_label[2,:]
fp = all_label[3,:]

m1 = sm.confusion_matrix(gt, gp)
m2 = sm.confusion_matrix(ft,fp)
label_strlist_1 = ['TIP', 'TMP','TRP', 'KP']
label_strlist_2 = ['10% MVC', '40% MVC', '70% MVC']
plot_confusion_matrix(m1, label_strlist_1,
                          'Confusion Matrix for Grip Gestures at 3 Force levels')
plt.savefig('C:\\Users\\86156\\Desktop\\fc3_1.png')
print(1)







