import pandas as pd
import numpy as np

#对MVC数据进行平滑滤波
def smooth_filter(data, alpha=0.1):
    """平滑滤波"""
    for i in range(1, len(data)):
        last = data[i-1]
        current = data[i]
        data[i] = last * (1.0 - alpha) + current * alpha
    return data

#计算MVC
path = 'C:\\Users\\86156\\Desktop\\PPG_experiment\\force_data\\MVC_sub2\\'

Total_MVC = []
MVC_list = []
for i in range(1,16):
    raw_data = pd.read_table(path + str(i) + '.txt', header=None, encoding='UTF-16 LE')
    raw_data = raw_data.iloc[2:, :]
    data = raw_data.to_numpy(dtype=np.float32)
    data = np.ravel(data)
    data = list(data)
    data = smooth_filter(data)
    number = np.mean(data)
    MVC_list.append(number)
    if i%3==0:
        MVC = sum(MVC_list) / len(MVC_list)
        MVC_list = []
        MVC_gesture = [MVC, 0.1*MVC, 0.4*MVC, 0.7*MVC]
        Total_MVC.append(MVC_gesture)

Total_MVC = np.array(Total_MVC)
print(Total_MVC)
np.savetxt('./MVC/sub2.csv', Total_MVC, delimiter=',')





