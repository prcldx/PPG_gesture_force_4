import numpy as np
from Tool_Pretreatment import *
import matplotlib.pyplot as plt
from Tool_Extractfeature import statistical
from User_modify import CONFIGURATION
import pandas as pd
import matplotlib.pyplot as plt
from Tool_Preprocess import *
from scipy.fftpack import fft,ifft
from sklearn.preprocessing import StandardScaler
from Tool_Pretreatment import *
from Tool_Visualization import *
from Tool_FFT import *




def produce_sample_label(path):

    """ 参数预定义 """
    HZ = 200 #原始数据的采样频率为200HZ
    init_length = 200
    path = path
    movedata_filter = [] #存放滤波后数据列表

    """ 读取原始数据 """
    movedata_0, samplingpoint_num_list = read_data(path)


    """ 数据清洗-校准-滤波 """

    for data in movedata_0:
        data = pd.DataFrame(data)
        data = data.iloc[0:1000,:] #把都进来的数据的长度限制在5s
        data = data.dropna()  # 数据清洗

        #data = data.iloc[init_length:, :]  # 手动去除PPG头部抖动的数据
        data = np.array(data)
        data = data[:, 0:-1] #不需要最后一列数据，最后一列数据为时间戳

        # 巴特沃斯滤波
        filtered_data = np.zeros([data.shape[0], 9])
        for filter in range(data.shape[1] - 1):
            if filter <= 2:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 1, fs=HZ)
                # filtered_data[:,filter] = butter_bandpass_filtfilt(data[:, filter], cutoff=[0.1, 1.0], fs=HZ)
            else:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 20, fs=HZ)

        #LMS
        # for i in range(3):
        #     filtered_data[:, i] = LMS_filter(dn=filtered_data[:, i], xn=filtered_data[:, [3, 4, 5]], mu=0.1)

        filtered_data = filtered_data[init_length:data.shape[0] - init_length, :]

        movedata_filter.append(filtered_data)


    """ 滑窗分段-加标签 """
    #预定义滑窗参数
    window_time = 0.2                   # Window length (s)
    window_len = int(window_time * HZ)     # Number of sample points in the window window_len=300
    ini_pass_len = 0          # Necessary. Need to discard the initial "ini_pass_len" sample points of the data stream
    increment = window_len/4
    
    #预定义数据存储列表
    movedata = []                                             # 存储分段后的数据
    label_pre = []                                            # 存储标签

    #预定义标签列表源
    subject_list = [0]
    hand_gesture = [0, 1, 2, 3]
    force_level = [0,1,2]
    trail = [0,1,2,3,4,5,6,7,8,9]
    window_number_list = []                                   # 列表：存储每一个data的窗的数量

    for data in movedata_filter:
        window_number = int((data.shape[0]-ini_pass_len-window_len)/increment+1)    #得到某个data分段的个数
        window_number_list.append(window_number)

        for k in range(window_number):

            med = ini_pass_len + int(increment)*k
            datapacket = data[med:med+int(window_len),:]
            movedata.append(datapacket)


    """ 对segment加标签"""
    for s in subject_list:
        for i in hand_gesture:
            for j in force_level:
                for k in trail:
                    l = i*3 + j #建立一个force 和 gesture 组合的label
                    label = [s, i, j, k, l] #建立标签列表
                    index = s*120 + i * 30 + j * 10 + k
                    for seg in range(window_number_list[index]):
                        label_pre.append(label)

    label_pre = np.array(label_pre)




    """ 分段提取特征"""

    # 定义特征提取参数
    # Specify the parameters related to extracting features
    sta_char_ppg = CONFIGURATION['ppg_feature']           #ppg feature

    movedata_feature = []
    for datapacket in movedata:
        feat_packet = []
        # Extract the statistical characteristics of acceleration (8 channels)

        for i in range(3):
            ppg_feat = statistical(datapacket[:, i], sta_char_ppg)
            feat_packet.append(ppg_feat)

        # Reduce the dimension of feat_packet to a one-dimensional list
        feat_packet = [element for element in feat_packet if element != []]     # Remove empty elements in feat_packet
        med_vari = str(feat_packet)
        med_vari = med_vari.replace('[', '')
        med_vari = med_vari.replace(']', '')
        med_vari = eval(med_vari)

        # Store feat_packet
        if type(med_vari) == tuple:
            feat_packet = list(med_vari)
        elif type(med_vari) == float:
            feat_packet = [med_vari]
        movedata_feature.append(feat_packet)
    movedata_feature = np.array(movedata_feature)
    return movedata_feature, label_pre








