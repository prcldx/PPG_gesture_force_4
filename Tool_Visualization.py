"""可视化模块"""

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt    # 绘图库
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

def dynamic_fig_single_row(data):
    """根据实时数据绘制动态图,一行
    数据流可视化"""
    plt.cla()  # 清除之前的图
    plt.ion()  # 打开交互模式(不阻塞)
    l1, = plt.plot(np.arange(len(data)), data,'r')
    plt.legend([l1], ['sin(x)'], loc='upper right')
    plt.xlim((0, 1000))  #设置纵轴范围
    plt.grid()          #开启网格
    plt.pause(0.01)     #暂停
    plt.show()
def plot_confusion_matrix(cm, labels_name, title):
    """绘制混淆矩阵"""
    plt.rcParams['font.size'] = 6 #应该建议为6
    plt.figure(figsize=(2.5, 2.5),dpi=800)#figure的大小要根据不同混淆矩阵的类型来 3.5 2.5 2
    ax = plt.gca()
    cm = cm.astype('float') / (cm.sum(axis=1).reshape(1,-1))    # 归一化

    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    #plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuGn)#GnBu
    #plt.title(title, fontsize=8 ,y=1.01)    # 图像标题
    cb1 =plt.colorbar(fraction=0.04, pad = 0.03)

    ax2 = cb1.ax
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(False)  # change width
    tick_locator = ticker.MaxNLocator(nbins=6)
    cb1.locator = tick_locator
    cb1.update_ticks()
    num_local = np.array(range(len(labels_name)))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)  # change width 改变整个图框的粗细

    plt.xticks(num_local, labels_name, fontsize=6, rotation = 60)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=6, rotation = 60)    # 将标签印在y轴坐标上
    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    cm = cm.T
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            a = cm[first_index][second_index]
            #b = "%.1f%%" % (a * 100)
            b = "%.1f" % (a * 100)
            if first_index == second_index: #and first_index < 5:
                plt.text(first_index, second_index, b, fontsize=12,  color="w", va='center', ha='center',weight = 'medium')

            else:
                plt.text(first_index, second_index, b, size=12, va='center', ha='center',weight = 'medium') #size应该为6
    #plt.subplots_adjust(bottom=0, top=0.05)
    plt.tight_layout()

def np_fig(length, data):
    """输入一维数组进行绘图"""
    plt.figure()
    plt.plot(np.arange(length), data)
    plt.show()


def df_row_fig(df, row_list, label_list, object_num, x_start, x_step_length, y_start, y_stop):
    """
    :param df: 输入的Data frame
    :param row_list: 所需要绘图的行 e.g.,[1310, 2710, 3710, 5110, 6160, 7360, 8860]
    :param label_list: 行对应的坐标['140[deg]','130[deg]','120[deg]','110[deg]','100[deg]','90[deg]','80[deg]']
    :param object_num: 点数
    :param x_start: X轴坐标起始点
    :param x_step_length: X轴坐标步长
    :param y_start: Y轴坐标起始点
    :param y_stop: Y轴坐标终止点
    :return:
    """
    plt.figure()
    my_x_ticks = np.arange(x_start, x_start+object_num*x_step_length, x_step_length)
    for i in range(0, len(row_list)):
        subject_draw = np.array(df.loc[row_list[i], 1:object_num])
        plt.plot(my_x_ticks, subject_draw, label=label_list[i])
    plt.legend()
    plt.ylim(y_start, y_stop)
    plt.ylabel()
    plt.xlabel()
    # plt.xticks(my_x_ticks)
    plt.show()


def df_column_fig(df, column_list, label_list, object_num, x_start, x_step_length, y_start, y_stop):
    """
    :param df:输入的Data frame
    :param column_list:所需要绘图的列 E.g.,['ACC_X','ACC_Y','ACC_Z','GYR_X','GYR_Y','GYR_Z','BIO']
    :param label_list:列对应的坐标
    :param object_num:点数
    :param x_start:X轴坐标起始点
    :param x_step_length:X轴坐标步长
    :param y_start:Y轴坐标起始点
    :param y_stop:Y轴坐标终止点
    :return:
    """
    plt.figure()
    my_x_ticks = np.arange(x_start, x_start+object_num*x_step_length, x_step_length)
    for i in range(0, len(column_list)):
        subject_draw = np.array(df.iloc[column_list[i], 1:object_num])
        plt.plot(my_x_ticks, subject_draw, label=label_list[i])
    plt.legend()
    plt.ylim(y_start, y_stop)
    plt.ylabel()
    plt.xlabel()
    # plt.xticks(my_x_ticks)
    plt.show()



def read_csv_to_df(path,index_list):
    """读取CSV文件，转成特定索引的dataframe"""


def transform_df_to_csv(df, index_list, name, path):
    """选取df的特定行列转成csv"""

def Visualization_PPG(x, data):
    plt.subplot(311)
    plt.plot(x, data[:,0], 'r')
    plt.title('R', fontsize=9, color='red')
    plt.subplot(312)
    plt.plot(x, data[:,1], 'blue')
    plt.title('IR', fontsize=9, color='blue')
    plt.subplot(313)
    plt.plot(x, data[:,2], 'green')
    plt.title('G', fontsize=9, color='green')
    plt.show()