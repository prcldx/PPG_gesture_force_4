""" 使用深度学习的方法进行分类 进行lot验证"""
import tensorflow as tf
from sklearn import preprocessing
import sklearn.metrics as sm
import numpy as np
from Tool_Pretreatment import *
import itertools
from DL_model import *


print(tf.__version__)



def get_batch(data, label, batch_size, batch_index, shulff=True):
    # 从数据集中"随机"取出batch_size个元素并返回
    if shulff:
        index = range(batch_index * batch_size, (batch_index + 1) * batch_size)
    else:
        index = np.random.randint(0, data.shape[0], batch_size)  # 这个是找出随机个数个元素的索引

    return data[index, :], label[index]

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

#在使用CNN时需要用到数据增强
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

sub_num = 1
fold = 10
HZ=200
CNN_flag = False


for sub_index in range(1,1+sub_num):
    path = './data/sub' + str(sub_index) + '/'
    dataset, label = load_data(path)

    #数据增强，增强之后用CNN再做,是一种数据加强的方式
    if CNN_flag:
        dataset = data_extension(dataset)

    #进行lot验证
    for trail_index in range(1):
        train_index = label[:,3] != trail_index
        test_index = label[:,3] == trail_index

        #划分训练集和测试集
        train_data = dataset[train_index,:]
        train_label = label[train_index]

        test_data = dataset[test_index,:]
        test_label = label[test_index]

        train_label = train_label[:,4]
        test_label = test_label[:,4]

        # 如果使用CNN的话，对data进行扩维处理
        if CNN_flag:
            train_data = np.expand_dims(train_data, axis=-1)
            test_data = np.expand_dims(test_data, axis=-1)

        # 模型参数设置

        epochs = 10
        batchsize = 100

        #建立模型

        #model = PPG_MLP()
        #model = PPG_CNN()
        #model = FUS(rank = 2 , output_dim = 15, dropouts = [0.2, 0.2, 0.2, 0.2],fnn_output=15 ,use_softmax = True )
        model = PPG_LSTM()

        optimizer = tf.keras.optimizers.Adam()

        for epoch_index in range(epochs):

            #将训练数据提前打乱
            index = np.arange(train_data.shape[0])
            np.random.shuffle(index)
            train_data = train_data[index]
            train_label = train_label[index]

            num_batches = int(train_data.shape[0] // batchsize)

            sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 定义评估器
            categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
            for batch_index in range(num_batches):
                X, y = get_batch(train_data, train_label, batch_size=batchsize, batch_index=batch_index)

                with tf.GradientTape() as tape:
                    #y_pred = model(X)
                    y_pred = model.forward(X)
                    #y_pred = model.forward(X[:,:,0], X[:,:,1], X[:,:,2])  # 注意观察y_pred的输出
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                    loss = tf.reduce_mean(loss)
                    #print(" epoch %d , batch %d: loss %f" % (epoch_index, batch_index, loss.numpy()))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, model.trainable_variables))  # 记得检验一下trainable_variables

                sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)

            #print("第 {} 次， 训练集准确率：{}".format(epoch_index,sparse_categorical_accuracy.result()))
            #使用验证集评估准确率
            validation_sc_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            #validate_y_pred = model(test_data)
            validate_y_pred = model.forward(test_data)
            #validate_y_pred = model.forward(test_data[:,:,0], test_data[:,:,1], test_data[:,:,2])
            validation_sc_accuracy.update_state(y_true=test_label, y_pred=validate_y_pred)
            print("第 {} 次， 训练集准确率：{}， 测试集准确率：{}".format(epoch_index, sparse_categorical_accuracy.result(), validation_sc_accuracy.result()))

            validation_sc_accuracy.reset_state()
            sparse_categorical_accuracy.reset_state()








