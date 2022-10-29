

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

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = 'Arial'


# 制作数据
x = np.linspace(0, 2 * np.pi, 100)
y1, y2 = np.sin(x), np.cos(x)

plt.plot(x, y1)
plt.plot(x, y2)

plt.title('xxxx')  # 标题
plt.xlabel('x')  # 横坐标
plt.ylabel('y')  # 纵坐标
plt.legend(['x1', 'x2'])  # 折线标签
plt.savefig('./img_test.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('./img_test.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.show()  # 即刻展示
