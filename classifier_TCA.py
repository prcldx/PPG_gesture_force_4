""" 在不同对象上迁移学习"""

""" 将总体作为研究对象"""
import pandas as pd
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

sub_num = 2
sub_score_array = []

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker=='primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        :param kernel_type:
        :param dim:
        :param lamb:
        :param gamma:
        """
        self.kernel_type = kernel_type  #选用核函数的类型
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        :param Xs: 源域的特征矩阵 （样本数x特征数）
        :param Xt: 目标域的特征矩阵 （样本数x特征数）
        :return: 经过TCA变换后的Xs_new,Xt_new
        """
        X = np.hstack((Xs.T, Xt.T))     #X.T是转置的意思；hstack则是将数据的相同维度数据放在一起
        X = X/np.linalg.norm(X, axis=0)  #按照列向量求L2范数
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        #构造MMD矩阵 L
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')#F范数
        #构造中心矩阵H
        H = np.eye(n) - 1 / n*np.ones((n, n))
        #构造核函数矩阵K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        #注意核函数K就是后边的X特征，只不过用核函数的形式表示了
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        b = np.linalg.multi_dot([K, H, K.T])#XHX_T

        w, V = scipy.linalg.eig(a, b)  #这里求解的是广义特征值，特征向量
        ind = np.argsort(w)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到ind
        A = V[:, ind[:self.dim]]#取前dim个特征向量得到变换矩阵A，按照特征向量的大小排列好,
        Z = np.dot(A.T, K)#将数据特征*映射A
        Z /= np.linalg.norm(Z, axis=0)#单位向量话
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#得到源域特征和目标域特征
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt , then make p r edi c tion s on ta rg e t using 1NN
        param Xs : ns ∗ n_feature , source feature
        param Ys : ns ∗ 1 , source label
        param Xt : nt ∗ n_feature , target feature
        param Yt : nt ∗ 1 , target label
        return : Accuracy and predicted_labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)#经过TCA映射
        clf = KNeighborsClassifier(n_neighbors=1) #k近邻分类器，无监督学习
        clf.fit(Xs_new, Ys.ravel())#训练源域数据
        # 然后直接用于目标域的测试
        y_pred = clf.predict(Xs_new)
        acc = sklearn.metrics.accuracy_score(Ys, y_pred)
        print(acc)
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred



# for sub_index in range(1, sub_num + 1):
#
#     path1 = './figure5_data/data' + str(sub_index) + '2.csv'
#     path2 = './figure5_data/label' + str(sub_index) + '2.csv'
#
#     dataset = pd.read_csv(path1, header=None)
#     label = pd.read_csv(path2, header=None)
#     dataset = np.array(dataset)
#     label = np.array(label)
#     label = label.astype(int)
#
#     # 打乱数据，为各个训练器准备训练数据
#     data_pool = np.zeros([dataset.shape[0], dataset.shape[1] + label.shape[1]])
#     data_pool[:, 0:dataset.shape[1]] = dataset
#     data_pool[:, dataset.shape[1]:] = label
#     np.random.seed(0)  # 控制随机数产生的种子
#     np.random.shuffle(data_pool)
#     dataset = data_pool[:, 0:dataset.shape[1]]
#     label = data_pool[:, dataset.shape[1]:]
#     label = label.astype(int)
#
#     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dataset)
#     dataset = scaler.transform(dataset)
#
#
#     if sub_index ==1:
#         train_data = dataset
#         train_label = label[:,4]
#     elif sub_index ==2:
#         test_data = dataset
#         test_label = label[:,4]





# 构造分类器的集合




knn = KNeighborsClassifier()  # KNN Modeling
clf = svm.SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators=16)  # RF Modeling
lda = LinearDiscriminantAnalysis()
selected_model = [rf, lda, knn, clf]

TCA_flag = 1
#建立TCA类
tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)

if TCA_flag==1:
    Xs_new, Xt_new = tca.fit(train_data, test_data)

model = rf
model.fit(Xs_new, train_label)  # 训练源域数据


# 然后直接用于目标域的测试
y_pred = clf.predict(Xs_new)
acc = sklearn.metrics.accuracy_score(train_label, y_pred)
print(acc)

#target domain
y_pred = model.predict(Xt_new)
acc = sklearn.metrics.accuracy_score(test_label, y_pred)
print(acc)

print(1)











