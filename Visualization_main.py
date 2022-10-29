""" 对原始的信号以及信号的特征进行监测"""
#对信号的特征进行显示
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
from Tool_Pretreatment import *

path1 = './Feature/data23.csv'
path2 = './Feature/label23.csv'
flag=3
"""
flag=1绘制ACC特征的


"""
HZ = 200

if flag==1:
    dataset = pd.read_csv(path1, header=None)
    dataset = np.array(dataset)
    labels_name = ['X_MA', 'X_WL', 'Y_MA', 'Y_WL', 'Z_MA', 'Z_WL']
    x = np.arange(dataset.shape[0])
    for i in range(6):
        plt.plot(x, dataset[:, i], label=labels_name[i])
    plt.legend()
    plt.show()

if flag==0:
    path = './data/sub11/'
    movedata_0, samplingpoint_num_list = read_data(path)
    init_length = 100
    movedata_1 = []
    for data in movedata_0:
        data = pd.DataFrame(data)
        data = data.iloc[0:1000,:] #把都进来的数据的长度限制在5s
        data = data.dropna()  # 数据清洗

        #data = data.iloc[init_length:, :]  # 手动去除PPG头部抖动的数据
        data = np.array(data)
        data = data[:, 0:-1] #不需要最后一列数据以及角速度计的数据，最后一列数据为时间戳
        #movedata_1.append(data)

        # 巴特沃斯滤波
        filtered_data = np.zeros([data.shape[0], 9])
        for filter in range(data.shape[1] - 1):
            if filter <= 2:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 1, fs=HZ)
                # filtered_data[:,filter] = butter_bandpass_filtfilt(data[:, filter], cutoff=[0.1, 1.0], fs=HZ)
            else:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 20, fs=HZ)

        filtered_data = filtered_data[init_length: data.shape[0]-init_length,:] #截断开头和结尾的数据
        movedata_1.append(filtered_data)


    movedata_1 = np.array(movedata_1)
    movedata_2 = np.zeros((movedata_1.shape[0]*movedata_1.shape[1],movedata_1.shape[2]))

    for index in range(movedata_1.shape[0]):
        data = movedata_1[index,:,:]
        movedata_2[movedata_1.shape[1]*index:movedata_1.shape[1]*(index+1),:] = data

    dataset = movedata_2[:,0:3]

    #x = np.arange(dataset.shape[0])
    x1 = np.arange(24000)
    x2 = np.arange(dataset.shape[0])
    Visualization_PPG(x2, dataset)
    for i in range(4):
        Visualization_PPG(x1, dataset[i*24000:(i+1)*24000,:])

    # labels_name = ['R','IR','G','X','Y','Z']
    # for i in range(2,3):
    #     plt.plot(x, dataset[0:24000, i], label=labels_name[i])
    # plt.legend()
    # tick = np.arange(0,24000+8000,8000)
    # plt.xticks(tick, fontsize = 6)
    plt.show()
    print(1)


if flag==3:
    path = './data/sub11/'
    movedata_0, samplingpoint_num_list = read_data(path)
    init_length = 100
    movedata_1 = []
    for data in movedata_0[0:1]:
        data = pd.DataFrame(data)
        data = data.iloc[0:1000,:] #把都进来的数据的长度限制在5s
        data = data.dropna()  # 数据清洗

        #data = data.iloc[init_length:, :]  # 手动去除PPG头部抖动的数据
        data = np.array(data)
        data = data[:, 0:-1] #不需要最后一列数据以及角速度计的数据，最后一列数据为时间戳
        #movedata_1.append(data)

        Visualization_PPG(np.arange(data.shape[0]), data[:,0:3])

        # 巴特沃斯滤波
        filtered_data = np.zeros([data.shape[0], 9])
        for filter in range(data.shape[1] - 1):
            if filter <= 2:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 1, fs=HZ)
                # filtered_data[:,filter] = butter_bandpass_filtfilt(data[:, filter], cutoff=[0.1, 1.0], fs=HZ)
            else:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 20, fs=HZ)

        filtered_data = filtered_data[init_length: data.shape[0]-init_length,:] #截断开头和结尾的数据
        Visualization_PPG(np.arange(filtered_data.shape[0]), filtered_data[:,0:3])

        movedata_1.append(filtered_data)

