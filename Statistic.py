import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
""" 几种分类器分类的准确率对比"""
# x = np.arange(4)
# print(x)
# y = [0.933, 0.934, 0.942, 0.975]
#
#
# plt.bar(x, y, alpha=0.5, width=0.5, color='green', lw=3)
# plt.legend(loc='upper left')
# plt.errorbar(x,y,yerr=error,fmt='.',ecolor='black',
# 			elinewidth=1,ms=5,mfc='wheat',mec='salmon',capsize=2)
#
# labels_name = ['KNN','SVM','RF','LDA']
# plt.xticks(x, labels_name, fontsize=20)  # 将标签印在x轴坐标上
# plt.ylabel('Classification Accuracy',fontsize=20)
#
# num_local = [0,0.2,0.4,0.6,0.8,1]
# ylabels = ['0','20%','40%','60%','80%','100%']
# plt.yticks(num_local, ylabels, fontsize=10)  # 将标签印在y轴坐标上
#
# for a, b in zip(x, y):
# 	plt.text(a, b + 0.001, '%.1f%%' % (b*100), ha='center', va='bottom', fontsize=16)
# plt.show()


path = './dd.xlsx'
data = pd.read_excel(path)
data = np.array(data)

PPG_IMU = data[0,:]
PPG = data[1,:]
R = data[2,:]
IR = data[3,:]
G = data[4,:]
print(1)

std_PI = np.std(PPG_IMU,ddof = 1)
std_PPG = np.std(PPG,ddof = 1)
std_R = np.std(R,ddof = 1)
std_IR = np.std(IR,ddof = 1)
std_G = np.std(G,ddof = 1)

x = np.arange(5)
error = [std_PI,std_PPG,std_R,std_IR,std_G]
y=[]
for i in range(5):
    mea = np.mean(data[i,:])
    y.append(mea)

#绘制误差图
plt.bar(x, y, alpha=0.5, width=0.2, color='green', lw=3)
#plt.legend(loc='upper left')
plt.errorbar(x, y,yerr=error,fmt='.',ecolor='black',
			elinewidth=1,ms=5,mfc='wheat',mec='salmon',capsize=5)

labels_name = ['PPG+IMU','PPG','R','IR','G']
plt.xticks(x, labels_name, fontsize=20)  # 将标签印在x轴坐标上
plt.ylabel('Classification Accuracy',fontsize=20)
for a, b in zip(x, y):
	plt.text(a, b + 0.02, '%.1f%%' % (b*100), ha='center', va='bottom', fontsize=16)
plt.show()

#绘制综合分类精度的对比图



matplotlib.rcParams['font.family'] = 'SimHei'
labels_name = ['Subject1','Subject2']
x = np.arange(2)
total_acc = [0.72, 0.63]
gesture_acc = [0.96, 0.82]


plt.bar(x, total_acc, label="Gesture & Force",width = 0.2 )
plt.bar(x+0.2, gesture_acc, width=0.2, label="Only Gesture",color = 'gold')
plt.xticks(x+0.1, labels_name,fontsize = 14)

plt.title("Classification Accuracy", fontsize = 14)
plt.legend(loc='upper right')
for a, b in zip(x, total_acc):
	plt.text(a, b, '%.1f%%' % (b*100), ha='center', va='bottom', fontsize=10)

for a, b in zip(x, gesture_acc):
	plt.text(a+0.2, b, '%.1f%%' % (b*100), ha='center', va='bottom', fontsize=10)

plt.show()

Force_0 = np.load('Force_0.npy')
Force_1 = np.load('Force_1.npy')
Force_2 = np.load('Force_2.npy')

x = np.arange(Force_0.shape[0])
force_list = [Force_0, Force_1]
colorlist = ['deepskyblue','gold']
labellist = ['Session1', 'Session2']
for i in range(2):
    if i == 1:
        x = x + 0.2
    plt.bar(x, force_list[i], width=0.2, label = labellist[i], color = colorlist[i])
    for a, b in zip(x, force_list[i]):
        plt.text(a, b, '%.1f%%' % (b * 100), ha='center', va='bottom', fontsize=10)

labels_name = ['Gesture1','Gesture2','Gesture3','Gesture4','Gesture5']
plt.xticks(x-0.1, labels_name,fontsize = 12)
plt.title('Force classification Accuracy over different gestures')
plt.legend()
plt.show()

x = np.arange(Force_0.shape[0])
force_list = [Force_1, Force_2]
colorlist = ['deepskyblue','gold']
labellist = ['Subject1', 'Subject2']
for i in range(2):
    if i == 1:
        x = x + 0.2
    plt.bar(x, force_list[i], width=0.2, label = labellist[i], color = colorlist[i])
    for a, b in zip(x, force_list[i]):
        plt.text(a, b, '%.1f%%' % (b * 100), ha='center', va='bottom', fontsize=10)

labels_name = ['Gesture1','Gesture2','Gesture3','Gesture4','Gesture5']
plt.xticks(x-0.1, labels_name,fontsize = 12)
plt.title('Force classification Accuracy over different gestures')
plt.legend()
plt.show()