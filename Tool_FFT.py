import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl


def FFT_transfer(Fs, data):
    """
    :param Fs: 采样频率
    :param data: 一维时域数组
    :return: 频域坐标，fft变换结果
    """
    L = data.shape[0]
    N = int(np.power(2, np.ceil(np.log2(L))))
    fft_data = fft(data, N)
    fft_data = np.abs(fft_data)/N   #归一化处理
    fft_data = fft_data[range(int(N/2))]
    x = np.arange(N)*Fs/N
    x = x[range(int(N/2))]
    return x, fft_data

def FFT_transfer2(a):
    fft_a = fft(a)
    N = a.shape[0]  # N代表着DFT变换的点数2477

    x = np.arange(0, N)  # 时域坐标的表达
    X = np.arange(0, N)  # 频域坐标的表达df*N = 200

    half_X = X[0:int(N / 10)]  # 取半fs区间，因为DFT变换的共轭对称性

    abs_a = np.abs(fft_a)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_a = np.angle(fft_a)  # 取复数的角度
    normalization_a = abs_a / N  # 归一化处理（双边频谱）
    normalization_half_a = normalization_a[0:int(N / 10)]  # 由于对称性，只取一半区间（单边频谱）
    # FFT绘图
    plt.subplot(221)
    plt.plot(x, a)
    plt.title('raw signal')
    #
    plt.subplot(222)
    plt.plot(X, abs_a, 'r')
    plt.title('two_side amplitude', fontsize=9, color='red')
    #
    plt.subplot(223)
    plt.plot(X, normalization_a, 'g')
    plt.title('two_side amplitude_Normalization', fontsize=9, color='green')
    #
    plt.subplot(224)
    plt.plot(half_X, normalization_half_a, 'blue')
    plt.title('one_side amplitude_Normalization', fontsize=9, color='blue')
    plt.show()



"""
构建测试信号
"""
# Fs = 400
#
# x = np.arange(0,1,1/400)
# print(x.shape[0])
# y=7*np.sin(2*np.pi*50*x) + 5*np.sin(2*np.pi*75*x)+3*np.sin(2*np.pi*125*x)
# X = np.arange(0,400)
# plt.subplot(311)
# plt.plot(X,y)
# plt.subplot(312)
# plt.plot(x,y)
#
# x_label,fft_data = FFT_transfer(Fs, y)
# plt.subplot(313)
# plt.plot(x_label, fft_data)
# plt.show()
