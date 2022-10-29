import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd



ACC_score_list = np.load('./new_figure/figure_1.npy')



def ANOVA(df, alpha=0.05):
    """
    :param df: 传入的df是一个字典，key为水平,value为每一次的trial
    :param alpha: 传入置信度
    :return:
    """
    df = pd.DataFrame(df)
    df_melt = df.melt()
    df_melt.columns = ['Treat', 'Value']
    model = ols('Value~C(Treat)', data=df_melt).fit()
    anova_table = anova_lm(model, typ=2)
    print(anova_table)
    mc = MultiComparison(df_melt['Value'], df_melt['Treat'])
    tukey_result = mc.tukeyhsd(alpha=alpha)
    print(tukey_result)
    return tukey_result


def two_way_ANOVA(df, alpha=0.1):
    """
    aim:研究没有交互作用的双因素方差分析
    :param df: 传入的df是一个dataframe数组，横轴和纵轴分别是两个因素
    :param alpha: 传入置信度
    :return:
    """

    df1 = df.stack().reset_index().rename(columns={0: 'value'})
    model = ols('value~C(A) + C(B)', df1).fit()
    print(anova_lm(model))
    print(pairwise_tukeyhsd(df1['value'], df1['B'], alpha=alpha))  # 第一个必须是销量， 也就是我们的指标。这个很正常，两两比较的方法一般比较保守


# df = {'60':[90,92,88],'65':[97,93,92],'70':[96,96,93],'75':[84,83,88],'80':[84,86,82]}
# ANOVA(df)

hg = {'tip':[89.3, 78.0, 77.1, 61.8], 'TMP': [92.8, 77.9, 72.4, 75.8], 'trp': [81.5, 71.0, 68.3, 67.5], 'kp': [84.0, 74.2, 70.6, 61.3]}
hg = pd.DataFrame(hg)
hg.index = pd.Index(['0','1','2','3'], name = 'A')
hg.columns = pd.Index(['tip','tmp','trp','kp'],name = 'B')
two_way_ANOVA(hg)
print(1)



feature_num = 7
sub_num = 14
RA_flag =6
mean_score = []
#two_way_ANOVA分析变量模型和变量subject对最终分类精度的影响
if RA_flag ==0:
    PPG = ACC_score_list[:, range(sub_num)]
    PPG_R = ACC_score_list[:, range(sub_num, 2 * sub_num)]
    PPG_IR = ACC_score_list[:, range(2 * sub_num, 3 * sub_num)]
    PPG_G = ACC_score_list[:, range(3 * sub_num, 4 * sub_num)]
    PPG_RIR = ACC_score_list[:, range(4 * sub_num, 5 * sub_num)]
    PPG_RG = ACC_score_list[:, range(5 * sub_num, 6 * sub_num)]
    PPG_IRG = ACC_score_list[:, range(6 * sub_num, 7 * sub_num)]
    d_PPG = pd.DataFrame(PPG)
    d_PPG.index = pd.Index(['rf','lda','knn','svm'], name = 'A')
    d_PPG.columns = pd.Index(['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14'],name = 'B')
    two_way_ANOVA(d_PPG)






#测试不同的模型之间是否有显著差异ANOVA,以及方便得出平均值写论文
if RA_flag==1:
    PPG = ACC_score_list[:, range(sub_num)]
    PPG_R = ACC_score_list[:, range(sub_num, 2 * sub_num)]
    PPG_IR = ACC_score_list[:, range(2 * sub_num, 3 * sub_num)]
    PPG_G = ACC_score_list[:, range(3 * sub_num, 4 * sub_num)]
    PPG_RIR = ACC_score_list[:, range(4 * sub_num, 5 * sub_num)]
    PPG_RG = ACC_score_list[:, range(5 * sub_num, 6 * sub_num)]
    PPG_IRG = ACC_score_list[:, range(6 * sub_num, 7 * sub_num)]
    feature_set_cata = [PPG, PPG_R, PPG_IR, PPG_G, PPG_RIR, PPG_RG, PPG_IRG]
    for feature in feature_set_cata:
        mean_fea = np.mean(feature, axis=1)
        mean_score.append(mean_fea)
    mean_score = np.array(mean_score)
    print(1)
    PPG = PPG*100
    other_model = []
    # for i in range(3):
    #     other_model.extend(list(PPG[i,:]))
    #df = {'rf':list(PPG[0,[10,11]]), 'lda':list(PPG[1,[10,11]]),'knn':list(PPG[2,[10,11]]),'svm':list(PPG[3,[10,11]]),}
    df = {'rf':list(PPG[0,0:2]), 'lda':list(PPG[1,0:2]),'knn':list(PPG[2,0:2]),'svm':list(PPG[3,0:2]),}
    df = {'knn': list(PPG[2, 0:2]), 'svm': list(PPG[3, 0:2]), }
    #df = {'rf': list(PPG_R[0, :]), 'lda': list(PPG_R[1, :]), 'knn': list(PPG_R[2, :]), 'svm': list(PPG_R[3, :]), }
    ANOVA(df, alpha=0.1)
    print(1)


#验证信号的组合是否有用
if RA_flag == 2:
    PPG = ACC_score_list[:, range(sub_num)].ravel()
    PPG_R = ACC_score_list[:, range(sub_num, 2 * sub_num)].ravel()
    PPG_IR = ACC_score_list[:, range(2 * sub_num, 3 * sub_num)].ravel()
    PPG_G = ACC_score_list[:, range(3 * sub_num, 4 * sub_num)].ravel()
    PPG_RIR = ACC_score_list[:, range(4 * sub_num, 5 * sub_num)].ravel()
    PPG_RG = ACC_score_list[:, range(5 * sub_num, 6 * sub_num)].ravel()
    PPG_IRG = ACC_score_list[:, range(6 * sub_num, 7 * sub_num)].ravel()

    feature_set_cata = [PPG, PPG_R, PPG_IR, PPG_G, PPG_RIR, PPG_RG, PPG_IRG]
    #df = {'PPG':list(PPG), 'PPG_R':list(PPG_R), 'PPG_IR':list(PPG_IR), 'PPG_G':list(PPG_G), 'PPG_RIR':list(PPG_RIR), 'PPG_RG':list(PPG_RG), 'PPG_IRG':list(PPG_IRG)}
    df = {'PPG_RIR': list(PPG_RIR), 'PPG_RG': list(PPG_RG), 'PPG_IRG': list(PPG_IRG)}
    #df = {'PPG_R':list(PPG_R), 'PPG_IR':list(PPG_IR), 'PPG_G':list(PPG_G)}
    ANOVA(df, alpha=0.05)
    print(1)

if RA_flag == 3:
    #研究结果2：手势分类分类器的影响，
    sub_num = 14
    rf_score = np.load('./new_figure/sub_score_array_rf.npy')
    lda_score = np.load('./new_figure/sub_score_array_lda.npy')
    knn_score = np.load('./new_figure/sub_score_array_knn.npy')
    svm_score = np.load('./new_figure/sub_score_array_svm.npy')

    model_score_list = [rf_score, lda_score, knn_score, svm_score]
    gesture_acc_list = []
    for i in range(4):
        gesture_acc_list.append(model_score_list[i][:,0,0])
    gesture_acc_list = np.array(gesture_acc_list)
    d_PPG = pd.DataFrame(gesture_acc_list)
    d_PPG.index = pd.Index(['rf', 'lda', 'knn', 'svm'], name='A')
    d_PPG.columns = pd.Index(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'],
                             name='B')
    two_way_ANOVA(d_PPG)

if RA_flag==4:
    #研究结果3：研究力分类不同信号的影响
    sub_num =14
    ACC_overfeature_list = np.load('./new_figure/ACC_overfeature_list.npy')
    d_PPG = pd.DataFrame(ACC_overfeature_list)
    d_PPG.index = pd.Index(['PPG', 'R', 'IR', 'G'], name='A')
    d_PPG.columns = pd.Index(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'],
                             name='B')
    two_way_ANOVA(d_PPG)

if RA_flag==5:
    #研究结果3：研究手势类型对力分类的影响
    sub_num =14
    ACC_score_list = np.load('./new_figure/Force_score_list.npy')

    Force_score_list = []
    mean_force_arr = np.zeros((16,4))
    std_force_arr = np.zeros((16,4))

    for r in range(int(ACC_score_list.shape[0]/sub_num)):
        for c in range(4):
            data = ACC_score_list[range(r*sub_num, r*sub_num+sub_num),c]
            mean_force = np.mean(data)
            std_force = np.std(data,ddof=1)

            mean_force_arr[r,c] = mean_force
            std_force_arr[r,c] = std_force

    SVM_score = mean_force_arr[12:,:]

    d_PPG = pd.DataFrame(SVM_score)
    d_PPG.index = pd.Index(['PPG', 'R', 'IR', 'G'], name='A')
    d_PPG.columns = pd.Index(['TIP', 'TMP', 'TRP', 'KP'],
                             name='B')
    two_way_ANOVA(d_PPG)

if RA_flag==6:
    #研究结果3：研究手势类型对力分类的影响，双因素分析，人+手势类型
    sub_num =14
    ACC_score_list = np.load('./new_figure/Force_score_list.npy')

    Force_score_list = []
    mean_force_arr = np.zeros((16,4))
    std_force_arr = np.zeros((16,4))

    ppg_all = ACC_score_list[168: 168+14,:]

    d_PPG = pd.DataFrame(ppg_all)
    d_PPG.index = pd.Index(list(np.arange(14)), name='A')
    d_PPG.columns = pd.Index(['TIP', 'TMP', 'TRP', 'KP'],
                             name='B')
    two_way_ANOVA(d_PPG)
    print(1)