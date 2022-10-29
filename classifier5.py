"""对每个手势中的力进行分类 """
""" 对手势的分类精度进行了解"""
""" 都是用的 leave one trail 的验证方式"""


""" 根据多数文献的选择，使用K-fold Validation 使用新的串联分类器"""
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



path1 = './Feature/data42.csv'
path2 = './Feature/label42.csv'

dataset = pd.read_csv(path1, header= None)
label = pd.read_csv(path2, header = None)
dataset = np.array(dataset)
label = np.array(label)
label = label.astype(int)

# 打乱数据，为各个训练器准备训练数据
data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
data_pool[:, 0:dataset.shape[1]] = dataset
data_pool[:, dataset.shape[1]:] = label
np.random.seed(0)#控制随机数产生的种子
np.random.shuffle(data_pool)
dataset = data_pool[:, 0:dataset.shape[1]]
label = data_pool[:, dataset.shape[1]:]
label = label.astype(int)
# dataset = dataset[:,4]
# dataset = dataset.reshape(5400,-1)

""" 构造分类器的集合"""
knn = KNeighborsClassifier()                #KNN Modeling
clf = svm.SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators=16)       #RF Modeling
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]
force_score_classifier_array = np.zeros((4,6))
gesture_score_classifier_array = np.zeros((4,5))

# dataset = dataset[label[:, 2] == 2, :]
# label = label[label[:, 2] == 2, :]
for model_index in range(4):

    candidate_model_a = selected_model[model_index]
    candidate_model_b = selected_model[model_index]

    #注释掉的内容是可以同时算出 手势分类精度和力分类精度
    #force_score_kfold_array, gesture_score_kfold_array = k_fold_validate(dataset, label, candidate_model_a, candidate_model_b)
    #force_score_classifier_array[model_index,:] = force_score_kfold_array

    gesture_score_kfold_array=gesture_k_fold_validate(dataset, label, candidate_model_a)
    gesture_score_classifier_array[model_index,:] = gesture_score_kfold_array


print('手势分类准确率:\n',gesture_score_classifier_array)
print('手势分类的平均准确率\n', np.mean(gesture_score_classifier_array, axis=1))
#print('力分类的准确率:\n',force_score_classifier_array)






