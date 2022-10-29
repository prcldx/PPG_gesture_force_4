from sklearn.neighbors import KDTree
import numpy as np


#这个类也是只针对离散型变量的
class ReliefF_2(object):
    """Feature selection using data-mined expert knowledge.
    Based on the ReliefF algorithm as introduced in:
    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55
    """

    def __init__(self, n_neighbors=100, n_features_to_keep = 10):
        """Sets up ReliefF to perform feature selection.
        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.
        Returns
        -------
        None
        """

        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep

    def fit(self, X, y):
        """Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(X.shape[1])
        self.tree = KDTree(X)

        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(X[source_index].reshape(1, -1), k=self.n_neighbors + 1)

            # First match is self, so ignore it
            for neighbor_index in indices[0][1:]:
                similar_features = X[source_index] == X[neighbor_index]
                label_match = y[source_index] == y[neighbor_index]

                # If the labels match, then increment features that match and decrement features that do not match
                # Do the opposite if the labels do not match
                if label_match:
                    self.feature_scores[similar_features] += 1.
                    self.feature_scores[~similar_features] -= 1.
                else:
                    self.feature_scores[~similar_features] += 1.
                    self.feature_scores[similar_features] -= 1.

        self.top_features = np.argsort(self.feature_scores)[::-1]

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[:self.n_features_to_keep]]


"""
# 说明：特征选择方法一：过滤式特征选择（ReliefF算法）
# 思想：先用特征选择过程对初始特征进行"过滤"，然后再用过滤后的特征训练模型
# 时间：2019-1-16
# 问题：
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
import random


# 异常类
class ReliefError:
    pass



class Relief_3:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: k近邻的个数
        """
        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = pd.DataFrame()
        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
                new_data[one] = self.__data[one]
                # print '%s 是数值型' % one
            else:
                # print '%s 是离散型' % one
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    # 返回一个样本的k个猜中近邻和其他类的k个猜错近邻
    def get_neighbors(self, row):
        df = self.get_data()
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.__k])
        # print list(right_sim_two.sort_values().index[0:self.__k])
        types = list(set(df[df.columns[-1]]) - set([row_type]))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.__k])
        return right, wrong

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        # data = self.__data.drop(self.__feature[-1], axis=1)
        data = self.__data
        row = data.iloc[index]
        right = 0
        for one in NearHit.values()[0]:
            nearhit = data.iloc[one]
            if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                max_feature = data[feature].max()
                min_feature = data[feature].min()
                right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            else:
                right_one = 0 if row[feature] == nearhit[feature] else 1
            right += right_one
        right_w = round(right / self.__k, 2)

        wrong_w = 0
        # 样本row所在的种类占样本集的比例
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            # 种类one在样本集中所占的比例
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong_one = 0
            for i in NearMiss[one]:
                nearmiss = data.iloc[i]
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    wrong_one_one = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                wrong_one += wrong_one_one

            wrong = round(p_one / (1 - p_row) * wrong_one / self.__k, 2)
            wrong_w += wrong
        w = wrong_w - right_w
        return w

    # 过滤式特征选择
    def reliefF(self):
        sample = self.get_data()
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        num = 1
        for i in sample_index:    # 采样次数
            one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            for f in self.__feature[0:-1]:
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        return f_w.mean()

    # 返回最终选取的特征
    def get_final(self):
        f_w = pd.DataFrame(self.reliefF(), columns=['weight'])
        final_feature_t = f_w[f_w['weight'] > self.__t]
        # final_feature_k = f_w.sort_values('weight').head(self.__k)
        # print final_feature_k
        return final_feature_t

# 几种距离求解
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)

def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim

def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB, rowvar=0)[0][1]

"""
if __name__ == '__main__':
    data = pd.read_csv('西瓜数据集31.csv')[['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', '类别']]
    f = Relief(data, 1, 0.2, 2)
    # df = f.get_data()
    # print type(df.iloc[0])
    # f.get_neighbors(df.iloc[0])
    # f.get_weight('色泽', 6, 7, 8)
    f.reliefF()
    # f.get_final()

"""
# -*- coding: utf-8 -*-

import numpy as np
import random
from sklearn.neighbors import KDTree


class MultiReliefF(object):

    def __init__(self, n_neighbors=10, n_features_to_keep=10, n_selected=10):
        """
        初始化实例化对象
        :param n_neighbors: 最近邻个数
        :param n_features_to_keep: 选取特征相关统计量最大的数量
        """
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
        self.n_selected = n_selected

    def fit(self, X, y):
        """
        计算特征的相关统计量的大小
        :param X: 数据部分
        :param y: 标签部分
        :return: 返回特征相关统计量数值的列表
        """
        # 记录每个特征的相关统计量，并初始化为0
        self.feature_scores = np.zeros(X.shape[1])
        # 获得了KDTree类实例化对象，后面用这个对象获得每个随机样本的K个最近邻
        self.tree = KDTree(X)
        num = X.max(axis=0) - X.min(axis=0)

        # 在样本数量范围内进行不重复的随机采样self.n_selected次
        random_list = random.sample(range(0, X.shape[0]), self.n_selected)

        for source_index in random_list:
            w = np.zeros(X.shape[1])
            # 当前采用的是单位权重计算公式。由于多标签中标签之间可能有相关性，所以不能简单的拿单标签的去计算。
            # 也可以采用其他权重计算公式
            weight = np.sum(y[source_index]) / y.shape[1] #这一行是考虑了多标签的影响

            # 由于是多标签数据集，所以需要对每一个标签进行传统意义上的ReliefF查询，再对查询出的结果进行加权。
            for label_index in range(y.shape[1]):
                label_data = y[:, label_index]
                # 此时是标签下的每一个分类
                diff_a = np.zeros(X.shape[1])
                diff_b = np.zeros(X.shape[1])

                # 对每一个标签进行去重，根据这个标签拥有的类别数进行循环，找到随机样本在每一类中的K个最近邻
                for label in np.unique(label_data):
                    # 通过np.where方法找到所有当前类别的样本的索引
                    each_class_samples_index = np.where(label_data == label)[0]
                    # 调用KDTree方法找到最近邻
                    data = X[each_class_samples_index, :]
                    distances, indices = self.tree.query(
                        X[source_index].reshape(1, -1), k=self.n_neighbors + 1)
                    # 此时indices是每个标签下每个类别中的K个近邻,因为自己离自己最近，所以要从1开始
                    indices = indices[0][1:]
                    # 本次实验所采用的数据集是连续类型的，所以要采用距离计算
                    # 如果是离散类型，那就直接调np.equal方法
                    if label == label_data[source_index]:
                        diff_a = np.sum((X[indices] - X[source_index]) ** 2, axis=0) / num
                    else:
                        prob = len(each_class_samples_index) / X.shape[0]
                        # 异类样本的相关统计量计算需要再乘以异类样本占所有样本的比例
                        diff_b += prob * (np.sum((X[indices] - X[source_index]) ** 2, axis=0) / num)
                # 最后对每一个标签的计算结果进行加权，就得到了最终每个样本计算的最终的相关统计量
                w += weight * (diff_b - diff_a) / (self.n_neighbors * self.n_selected)
            self.feature_scores += w

        # 根据对象初始化时的值，返回靠前的一些特征组成的数据子集。
        self.top_features = np.argsort(self.feature_scores)[::-1]
        return X[:, self.top_features[:self.n_features_to_keep]]

