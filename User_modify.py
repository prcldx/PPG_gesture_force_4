"""
Users only need to modify this document to adjust some parameters in the initial_testing and data process programs to view the results of the classification,
without having to fully understand the details of the program.

'accx_feature': Feature types extracted from x-axis acceleration data.
'accy_feature': Feature types extracted from y-axis acceleration data.
'accz_feature': Feature types extracted from z-axis acceleration data.
'resultantacc_feature': Feature types extracted from resultant acceleration data.
'gyrox_feature': Feature types extracted from x-axis angular velocity data.
'gyroy_feature': Feature types extracted from y-axis angular velocity data.
'gyroz_feature': Feature types extracted from z-axis angular velocity data.
'resultantgyro_feature': Feature types extracted from resultant angular velocity data.
'pitch_feature': Feature types extracted from pitch angle. (Pitch angle is one of the Euler angles and is calculated from acceleration.)
'roll_feature': Feature types extracted from roll angle. (Roll angle is one of the Euler angles and is calculated from acceleration.)
'pressure_feature': Feature types extracted from air pressure data. One "RANGE" feature is enough.
The optional features are as follows: 均值、无偏标准差、偏度、峰度、均方根、平均绝对偏差、四分位差、整流平均值,波形因子,频谱峰值,频谱峰值频率,正负数据交替出现的次数. 趋势(仅仅用为气压数据的特征)
    Mean: 'MEAN'.
    Unbiased standard deviation: 'USD'.
    Skewness: 'SK'.
    Kurtosis: 'KU'.
    Root mean square: 'RMS'.
    Mean absolute deviation: 'MAD'.
    Interquartile range: 'IR'.
    Rectified mean: 'RM'.
    Waveform factor: 'WF'.
    Spectral peak: 'SP'.
    Spectrum peak frequency: 'SPF'
    Number of alternating occurrences of positive or negative values: 'APN'.
    Trend (specially for air pressure data, extracting this feature from data of other types of sensors is useless): 'TREND'.
Please strictly follow the format below to modify the parameters:
    If you don't want any features for the data of one certain sensor, use [] or ''.
    If you want one feature, use ['MEAN'].
    If you want several features, use ['MEAN', 'USD'].
    (The order of the features has been determined by the program and has nothing to do with the order of the string flags entered by the user.)
    If you want all the features (not including 'TREND') for acc or gyro data, please use 'ALL'.
    Note that for air pressure data, the "TREND" feature is enough. The air pressure data does not have more information.

'model': Machine learning model for classification.
The optional models are as follows: K最近邻, 支持向量机, 随机森林, 线性判别分析，人工神经网络，高斯贝叶斯
    K-NearestNeighbor: 'KNN'.
    Support vector machines: 'SVM'.
    Random Forest: 'RF'.
    Linear Discriminant Analysis: 'LDA'.
    Artificial Neural Network: 'MLP'.
    TrAdaboost: 'TrA' (only for Transfer Task).
Please strictly follow the format below to modify the parameter:
    Can not be empty.
    If you want one model in Task 1 or 2,, use ['KNN'].
    If you want all models in Task 1 or 2, use ['KNN', 'SVM', 'RF', 'LDA', 'GNB'].
    For Transfer Task, you can use ['KNN', 'SVM', 'RF', 'LDA', 'GNB', 'TrA'] to compare performance of ML (Machine Learning) and TL (Transfer Learning) models.
"""

# CONFIGURATION = {
#                  'ppg_feature': 'ALL',                            # feature set
#                  'accx_feature': '',
#                  'accy_feature': '',
#                  'accz_feature': '',
#                  'resultantacc_feature': '',
#                  'gyrox_feature': '',
#                  'gyroy_feature': '',
#                  'gyroz_feature': '',
#                  'resultantgyro_feature': '',
#                  'pitch_feature': ['APN'],
#                  'roll_feature': ['APN'],
#                  'model': ['KNN','SVM','RF','LDA']}        # 'KNN', , 'LDA'

CONFIGURATION = {
                 'ppg_feature': 'TD',                            # feature set
                 'accx_feature': ['MAV', 'WL'],
                 'accy_feature': ['MAV', 'WL'],
                 'accz_feature': ['MAV', 'WL'],
                 'resultantacc_feature': ['WF', 'MAD', 'IR', 'MEAN', 'RM', 'USD', 'RMS'],
                 'gyrox_feature': '',
                 'gyroy_feature': '',
                 'gyroz_feature': '',
                 'resultantgyro_feature': '',
                 'pitch_feature': ['APN'],
                 'roll_feature': ['APN'],
                 'model': ['KNN','SVM','RF','LDA']}        # 'KNN', , 'LDA'


# CONFIGURATION = {
#                  'ppg_feature': 'ALL',                            # feature set
#                  'accx_feature': ['IR','MAD', 'USD'],
#                  'accy_feature': ['WF'],
#                  'accz_feature': ['IR', 'USD', 'MAD'],
#                  'resultantacc_feature': ['WF', 'MAD', 'IR', 'MEAN', 'RM', 'USD', 'RMS'],
#                  'gyrox_feature': '',
#                  'gyroy_feature': '',
#                  'gyroz_feature': '',
#                  'resultantgyro_feature': '',
#                  'pitch_feature': ['APN'],
#                  'roll_feature': ['APN'],
#                  'model': ['KNN','SVM','RF','LDA']}        # 'KNN', , 'LDA'
