""" 使用深度学习的方法进行分类"""
import tensorflow as tf
from sklearn import preprocessing
import sklearn.metrics as sm
import numpy as np
from Tool_Pretreatment import *
import itertools
from DL_model import *

print(tf.__version__)

sub_num = 2
fold = 10
HZ=200


def load_data(path):
    movedata_0, samplingpoint_num_list = read_data(path)
    init_length = 200
    movedata_filter = []
    for data in movedata_0:
        data = pd.DataFrame(data)
        data = data.iloc[0:1000, :]  # 把都进来的数据的长度限制在5s
        data = data.dropna()  # 数据清洗

        data = data.iloc[init_length:, :]  # 手动去除PPG头部抖动的数据
        data = np.array(data)
        data = data[:, 0:-1]  # 不需要最后一列数据以及角速度计的数据，最后一列数据为时间戳
        # movedata_1.append(data)

        # 巴特沃斯滤波
        filtered_data = np.zeros([data.shape[0], 9])
        for filter in range(data.shape[1] - 1):
            if filter <= 2:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 20, fs=HZ)
                # filtered_data[:,filter] = butter_bandpass_filtfilt(data[:, filter], cutoff=[0.1, 1.0], fs=HZ)
            else:
                filtered_data[:, filter] = butter_lowpass_filter(data[:, filter], 20, fs=HZ)

        movedata_filter.append(filtered_data)

    """ 滑窗分段-加标签 """
    # 预定义滑窗参数
    window_time = 0.2  # Window length (s)
    window_len = int(window_time * HZ)  # Number of sample points in the window window_len=300
    ini_pass_len = 0  # Necessary. Need to discard the initial "ini_pass_len" sample points of the data stream
    increment = window_len / 4

    # 预定义数据存储列表
    movedata_1 = []  # 存储分段后的数据
    label_pre = []  # 存储标签

    # 预定义标签列表源
    subject_list = [0]
    hand_gesture = [0, 1, 2, 3, 4]
    # hand_gesture = [0,1,2,3,4,5,6,7]
    force_level = [0, 1, 2]
    trail = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    window_number_list = []  # 列表：存储每一个data的窗的数量

    for data in movedata_filter:
        window_number = int((data.shape[0] - ini_pass_len - window_len) / increment + 1)  # 得到某个data分段的个数
        window_number_list.append(window_number)

        for k in range(window_number):
            med = ini_pass_len + int(increment) * k
            datapacket = data[med:med + int(window_len), :]
            movedata_1.append(datapacket)

    """ 对segment加标签"""
    for s in subject_list:
        for i in hand_gesture:
            for j in force_level:
                for k in trail:
                    l = i * 3 + j  # 建立一个force 和 gesture 组合的label
                    label = [s, i, j, k, l]  # 建立标签列表
                    index = s * 150 + i * 30 + j * 10 + k
                    for seg in range(window_number_list[index]):
                        label_pre.append(label)

    label_pre = np.array(label_pre)
    label = label_pre

    movedata_1 = np.array(movedata_1)
    dataset = movedata_1[:, :, 0:3]

    for i in range(dataset.shape[0]):
        dataset[i] = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset[i])

    return dataset, label

def data_extension(dataset):
    new_dataset = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2] * 6))
    for i in range(dataset.shape[0]):
        p_array = [0, 1, 2]  # 生成全排列的序列
        pailie = list(itertools.permutations(p_array))  # 要list一下，不然它只是一个对象
        for p_index in range(len(pailie)):
            p = pailie[p_index]
            k = dataset[i, :, p]
            k = np.transpose(k)
            new_dataset[i, :, p_index * 3: (p_index + 1) * 3] = k

    dataset = new_dataset
    return dataset





acc_oversub = []
for sub_index in range(1,1+sub_num):
    path = './data/sub' + str(sub_index) + '/'
    dataset, label = load_data(path)

    #数据增强，增强之后用CNN再做
    dataset = data_extension(dataset)

    #进行lot验证
    acc_overtrail = []
    for trail_index in range(1):
        train_index = label[:,3] != trail_index
        test_index = label[:,3] == trail_index

        #划分训练集和测试集
        train_data = dataset[train_index,:]
        train_label = label[train_index]

        test_data = dataset[test_index,:]
        test_label = label[test_index]

        # 建立模型（mdoel1为一般的全连接层模型），每一次都建立全新的模型
        model1 = tf.keras.Sequential()
        model1.add(tf.keras.layers.Flatten(input_shape=(40, 3), dtype=tf.float32))  # 扁平成向量
        model1.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model1.add(tf.keras.layers.BatchNormalization()) #进行batch_normalization
        model1.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model1.add(tf.keras.layers.BatchNormalization()) #进行batch_normalization
        model1.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model1.add(tf.keras.layers.Dense(15, activation='softmax'))

        #使用简单的卷积神经网络
        model2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                                   input_shape=(40, 18, 1)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same'),
            # tf.keras.layers.Conv2D(filters=16,kernel_size=2,activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dense(84,activation='relu'),
            tf.keras.layers.Dense(15, activation='softmax')])

        model3 = RNN()
        model = model2

        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

        # 如果使用CNN的话，对data进行扩维处理
        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)

        # 对模型进行训练
        history = model.fit(train_data, train_label[:,4], epochs=50, batch_size=80)
        predicted_label = model.predict(test_data)
        predicted_label = np.argmax(predicted_label, axis=1)
        acc_score = sm.accuracy_score(test_label[:,4], predicted_label)
        acc_overtrail.append(acc_score)


    acc_overtrail = np.mean(acc_overtrail)
    acc_oversub.append(acc_overtrail)

print(acc_oversub)

# path1 = './Feature/data12.csv'
#     path2 = './Feature/label12.csv'
#
#     dataset = pd.read_csv(path1, header=None)
#     label = pd.read_csv(path2, header=None)
#     dataset = np.array(dataset)
#     label = np.array(label)
#     label = label.astype(int)
#
#     #打乱数据
#     data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
#     data_pool[:, 0:dataset.shape[1]] = dataset
#     data_pool[:, dataset.shape[1]:] = label
#     np.random.seed(0)  # 控制随机数产生的种子
#     np.random.shuffle(data_pool)
#     dataset = data_pool[:, 0:dataset.shape[1]]
#     label = data_pool[:, dataset.shape[1]:]
#     label = label.astype(int)
