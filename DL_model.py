import tensorflow as tf
from sklearn import preprocessing
import sklearn.metrics as sm
import numpy as np
from Tool_Pretreatment import *
import itertools

#待修改，掌握LSTM的运行逻辑
class PPG_LSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)#return_sequences = True，返回所有time_step 的output
        self.lstm2 = tf.keras.layers.LSTM(128)
        self.dense1 = tf.keras.layers.Dense(15)
        self.dense2 = tf.keras.layers.Dense(128)
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense2(x)
        output = self.dense1(x)
        output = tf.nn.softmax(output)
        return output

    def forward(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense2(x)
        output = self.dense1(x)
        output = tf.nn.softmax(output)
        return output


class PPG_MLP(tf.keras.Model):
    def __init__(self):
        super(PPG_MLP, self).__init__()

        self.layer1 = tf.keras.layers.Flatten(input_shape=(40, 3), dtype=tf.float32)
        self.layer2 = tf.keras.layers.Dense(128)
        self.layer3 = tf.keras.layers.Dense(128)
        self.layer5 = tf.keras.layers.Dense(15, activation='softmax')
        self.bt1 = tf.keras.layers.BatchNormalization()
        self.bt2 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.keras.layers.Activation('relu')
        self.ac2 = tf.keras.layers.Activation('relu')

    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.bt1(x)
        x = self.ac1(x)
        x = self.layer3(x)
        x = self.bt2(x)
        x = self.ac2(x)
        output = self.layer5(x)
        return output

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.bt1(x)
        x = self.ac1(x)
        x = self.layer3(x)
        x = self.bt2(x)
        x = self.ac2(x)
        output = self.layer5(x)
        return output



class PPG_CNN(tf.keras.Model):
    def __init__(self):
        super(PPG_CNN, self).__init__()

        self.cov1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='valid',
                                   input_shape=(40, 18, 1))
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='valid')
        self.droplayer1 = tf.keras.layers.Dropout(0.2)
        self.cov2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same')
        self.droplayer2 = tf.keras.layers.Dropout(0.2)
        self.flatlayer1 = tf.keras.layers.Flatten()
        self.denselayer1 = tf.keras.layers.Dense(120, activation='relu')
        self.denselayer2 = tf.keras.layers.Dense(15, activation='softmax')
        self.bt1 = tf.keras.layers.BatchNormalization()


    def call(self, input):
        x = self.cov1(input)
        x = self.droplayer1(x)
        x = self.pool1(x)

        x = self.cov2(x)
        x = self.droplayer2(x)
        x = self.pool2(x)

        x = self.flatlayer1(x)
        x = self.denselayer1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = self.denselayer2(x)

        return output

    def forward(self, input):
        x = self.cov1(input)
        x = self.pool1(x)
        x = self.droplayer1(x)
        x = self.cov2(x)
        x = self.pool2(x)
        x = self.droplayer2(x)
        x = self.flatlayer1(x)
        x = self.denselayer1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = self.denselayer2(x)

        return output


#建立预处理模型
class FNN(tf.keras.Model):
    def __init__(self, drop_prob, fnn_output):
        super(FNN, self).__init__()
        self.drop_prob = drop_prob
        self.fnn_output = fnn_output
        self.layer_batch = tf.keras.layers.BatchNormalization()
        self.layer_drop = tf.keras.layers.Dropout(drop_prob)
        self.layer_dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.layer_dense2 = tf.keras.layers.Dense(self.fnn_output, activation='softmax')
        self.layer_dense3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.layer_dense4 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    def call(self, inputs):
        #x = self.layer_batch(inputs)
        x = self.layer_dense1(inputs)
        x = self.layer_batch(x)
        x = self.layer_drop(x)
        x = self.layer_dense3(x)
        x = self.layer_dense4(x)
        output = self.layer_dense2(x)
        return output


class FUS(tf.keras.Model):
    def __init__(self, rank , output_dim, dropouts, fnn_output,use_softmax = False ):
        super(FUS, self).__init__()
        #接收传入的基本参数
        self.rank = rank #传入factor的秩
        self.use_softmax = use_softmax
        self.output_dim = output_dim
        self.fnn_output = fnn_output

        #定义每种信号的dropout概率
        self.ppg_r_prob = dropouts[0]
        self.ppg_ir_prob = dropouts[1]
        self.ppg_g_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        #生成每种信号初始的模型
        self.ppg_r_fnn = FNN(self.ppg_r_prob, self.fnn_output)
        self.ppg_ir_fnn = FNN(self.ppg_ir_prob, self.fnn_output)
        self.ppg_g_fnn = FNN(self.ppg_g_prob, self.fnn_output)



        # 生成融合之后的droupout层
        self.post_fusion_dropout = tf.keras.layers.Dropout(self.post_fusion_prob)

        #生成可训练的参数
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)#设置初始化生成器
        initializer_0 = tf.constant_initializer(0.)#设置0初始化器
        self.initial_value = initializer(shape=(self.rank, self.fnn_output + 1, self.output_dim))
        self.ppg_r_factor = tf.Variable(initial_value= self.initial_value)
        self.ppg_ir_factor = tf.Variable(initial_value=self.initial_value)
        self.ppg_g_factor = tf.Variable(initial_value=self.initial_value)
        self.fusion_weights = tf.Variable(initial_value=initializer(shape=(1, self.rank)))
        self.fusion_bias = tf.Variable(initial_value=initializer_0(shape=(1, self.output_dim)))#设置为浮点类型

    def forward(self, ppg_r_x, ppg_ir_x, ppg_g_x):

        #获取预处理后的tensor
        ppg_r_h = self.ppg_r_fnn(ppg_r_x)
        ppg_ir_h = self.ppg_ir_fnn(ppg_ir_x)
        ppg_g_h = self.ppg_g_fnn(ppg_g_x)

        batch_size = ppg_r_h.shape[0]

        #拼接tensor(最后一列全部拼接1)
        initializer_1 = tf.constant_initializer(1.)  # 设置1初始化器
        initial_ones = tf.Variable(initializer_1(shape=(batch_size,1)), trainable=False, dtype=tf.float32)#注意设置浮点型精度
        _ppg_r_h = tf.concat([ppg_r_h, initial_ones], axis=-1)
        _ppg_ir_h = tf.concat([ppg_ir_h, initial_ones], axis=-1)
        _ppg_g_h = tf.concat([ppg_g_h, initial_ones], axis=-1)

        #运算
        fusion_r = tf.matmul(_ppg_r_h, self.ppg_r_factor)
        fusion_ir = tf.matmul(_ppg_ir_h, self.ppg_ir_factor)
        fusion_g = tf.matmul(_ppg_g_h, self.ppg_g_factor)
        fusion_zy = fusion_r * fusion_ir * fusion_g #对应位置点对点相乘相加
        fusion_zy_per = tf.transpose(fusion_zy,perm=(1,0,2))
        output = tf.matmul(self.fusion_weights, fusion_zy_per)
        #重构张量的维度
        output_squeeze = tf.squeeze(output)
        output = output_squeeze+self.fusion_bias
        if self.use_softmax:
            output = tf.nn.softmax(output)
        return output



    def call(self, ppg_r_x, ppg_ir_x, ppg_g_x):

        #获取预处理后的tensor
        ppg_r_h = self.ppg_r_fnn(ppg_r_x)
        ppg_ir_h = self.ppg_ir_fnn(ppg_ir_x)
        ppg_g_h = self.ppg_g_fnn(ppg_g_x)

        batch_size = ppg_r_h.shape[0]

        #拼接tensor(最后一列全部拼接1)
        initializer_1 = tf.constant_initializer(1.)  # 设置1初始化器
        initial_ones = tf.Variable(initializer_1(shape=(batch_size,1)), trainable=False, dtype=tf.float32)#注意设置浮点型精度
        _ppg_r_h = tf.concat([ppg_r_h, initial_ones], axis=-1)
        _ppg_ir_h = tf.concat([ppg_ir_h, initial_ones], axis=-1)
        _ppg_g_h = tf.concat([ppg_g_h, initial_ones], axis=-1)

        #运算
        fusion_r = tf.matmul(_ppg_r_h, self.ppg_r_factor)
        fusion_ir = tf.matmul(_ppg_ir_h, self.ppg_ir_factor)
        fusion_g = tf.matmul(_ppg_g_h, self.ppg_g_factor)
        fusion_zy = fusion_r * fusion_ir * fusion_g #对应位置点对点相乘相加
        fusion_zy_per = tf.transpose(fusion_zy,perm=(1,0,2))
        output = tf.matmul(self.fusion_weights, fusion_zy_per)
        #重构张量的维度
        output_squeeze = tf.squeeze(output)
        output = output_squeeze+self.fusion_bias
        if self.use_softmax:
            output = tf.nn.softmax(output)
        return output





# x_a = np.random.rand(100,20)
# x_b = np.random.rand(100,20)
# x_c = np.random.rand(100,20)
#
# model = FUS(rank = 8 , output_dim = 15, dropouts = [0.2, 0.2, 0.2, 0.2], use_softmax = True )
# output = model.realize(x_a,x_b,x_c)
# print(output)
# print(1)













