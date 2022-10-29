""" 使用一般的机器学习模型进行 "综合分类" 的评估，出混淆矩阵的图"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 4})

ACC_score_list = pd.read_csv('./figure_note/ACC_score_list', header=None)
ACC_score_list = np.array(ACC_score_list)

feature_num = 7
RA_flag =1
if RA_flag==1:
    PPG = ACC_score_list[:, range(2)]
    PPG_R = ACC_score_list[:, range(2, 4)]
    PPG_IR = ACC_score_list[:, range(4, 6)]
    PPG_G = ACC_score_list[:, range(6, 8)]
    PPG_RIR = ACC_score_list[:, range(8, 10)]
    PPG_RG = ACC_score_list[:, range(10, 12)]
    PPG_IRG = ACC_score_list[:, range(12, 14)]
    print(1)
    feature_set_cata = [PPG, PPG_R, PPG_IR, PPG_G, PPG_RIR, PPG_RG, PPG_IRG]
    labels_name = ['RF', 'LDA', 'KNN', 'SVM']
    sensor_name = ['PPG-ALL', 'PPG-R', 'PPG-IR', 'PPG-G', ' PPG-R+IR','PPG-R+G', 'PPG-IR+G' ]
    color_list = ['skyblue', 'lightcoral', 'navajowhite', 'limegreen','peru','royalblue','m']
    plt.figure(figsize=(3.5,3), dpi=600)
    ax = plt.gca()
    bwith = 0.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    for i in range(feature_num):
        x = np.arange(4)*1.6
        y = np.mean(feature_set_cata[i], axis=1)
        error = np.std(feature_set_cata[i], axis=1, ddof=1)
        plt.bar(x + 0.2 * i, y * 100, alpha=0.6, width=0.2, lw=3, label=sensor_name[i], color=color_list[i])
        plt.errorbar(x + 0.2 * i, y * 100, yerr=error * 100, fmt='.', ecolor='black',
                     elinewidth=0.3, ms=0.1, mfc='wheat', mec='salmon', capsize=1, capthick=0.2)
    plt.xticks(x + 0.6, labels_name, fontsize=4)
    plt.xlabel('Machine learning Model', loc='center', fontsize=6, weight='medium')
    plt.ylabel('Recognition Accuracy (%) ', fontsize=6)
    plt.tick_params(width=0.5)#设置刻度线条的粗细
    plt.legend(loc = 'upper left',bbox_to_anchor = (0.05,-0.15) ,fontsize = 4, ncol = 5)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('./figure_final2.png')
    plt.show()

