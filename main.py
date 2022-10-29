"""
此代码为baseline程序
旨在获得相关特征矩阵储存在csv文件中
方便后续使用各种算法进行验证
dataset label 是只有ppg的特征
dataset2 label2 有ppg+imu的特征
dataset3 label3 把窗长由500ms改成了200ms
"""
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

""" 数据预处理(函数接口)"""
path = './data/yezeng/'
dataset, label  = produce_sample_label(path)

#将数据先写入csv文件，节省训练的时间
print(1)
np.savetxt('./Feature_2/data163.csv', dataset, delimiter=',')
np.savetxt('./Feature_2/label163.csv', label, delimiter=',')

# 31代表第3个实验对象的第1组结果 32就代表第3个实验对象的第2组结果

#
"""
31: 20HZ的低频滤波 40的窗长 步长为10
32: 1HZ的低频滤波 40的窗长 步长为10
33： 去掉SSC特征
34: 在33的基础上增加3个特征 skewness,kurtosis,interquartile_range
35: 在33的基础上增加2个频域特征
"""
















