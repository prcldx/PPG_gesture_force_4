import numpy as np
import scipy.io as sio
from scipy import signal
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


#定义巴特沃斯滤波器,输入的data是时域的函数
#axis=0表示跨行，axis=1表示跨列，作为方法动作的副词

def butter_bandpass_filtfilt(data,cutoff,fs=200.0,oder=4):
    wn = [2*i/fs for i in cutoff]
    b,a = signal.butter(oder,wn,'bandpass',analog=False)#滤波器构造函数
    output = signal.filtfilt(b,a,data,axis=0)#进行滤波
    return output


def butter_bandpass_filter(data, f1, f2, n, sample_f):
    """带通滤波器,注意要乘2"""
    b, a = signal.butter(n, [2*f1/sample_f, 2*f2/sample_f],'bandpass')
    #b, a = signal.butter(9, [2 * f2 / sample_f], 'lowpass')
    filter_data = signal.filtfilt(b, a, data)
    return filter_data