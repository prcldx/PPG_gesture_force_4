from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold
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

"""
利用网格搜索进行参数估计
"""



# Loading the Digits dataset
#digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target
#
# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#
# # Set the parameters by cross-validation
# tuned_parameters = [
#     {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
#     {"kernel": ["linear"], "C": [1, 1000]},
# ]

import numpy as np

test_label_oversub_list = np.load('./new_figure/test_label_oversub_list.npy')
predicted_label_oversub_list = np.load('./new_figure/predicted_label_oversub_list.npy')

m = sm.confusion_matrix(test_label_oversub_list, predicted_label_oversub_list)
print(m)
label_strlist = ['TIP10', 'TIP40', 'TIP70', 'TMP10', 'TMP40', 'TMP70', 'TRP10',
                             'TRP40',
                             'TRP70', 'KP10', 'KP40', 'KP70' ]
plot_confusion_matrix(m, label_strlist,
                                  'Confusion Matrix for Grip Gestures at 3 Force levels' )
plt.savefig('C:\\Users\\Lenovo\\Desktop\\New_Figure\\fc1.png')
plt.show()
print(1)