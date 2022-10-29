import numpy as np
# df = {'ctl':list(np.random.normal(10,5,100)),
#       'treat1':list(np.random.normal(15,5,100)),
#       'treat2':list(np.random.normal(20,5,100)),
#       'treat3':list(np.random.normal(30,5,100)),
#       'treat4':list(np.random.normal(31,5,100))}
# #组合成数据框
# import pandas as pd
# df = pd.DataFrame(df)
# df.head()
#
# df_melt = df.melt()
# df_melt.head()
#
# df_melt.columns = ['Treat','Value']
# df_melt.head()
#
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# model = ols('Value~C(Treat)',data=df_melt).fit()
# anova_table = anova_lm(model, typ = 2) #blog说typ=2是代表dataframe类型
# print(anova_table)
#
# from statsmodels.stats.multicomp import MultiComparison
# mc = MultiComparison(df_melt['Value'],df_melt['Treat'])
# tukey_result = mc.tukeyhsd(alpha = 0.05)
# print(tukey_result)

# def ANOVA(df,alpha=0.05):
#       """
#       :param df: 传入的df是一个字典，key为水平,value为每一次的trial
#       :param alpha: 传入置信度
#       :return:
#       """
#       df = pd.DataFrame(df)
#       df_melt = df.melt()
#       df_melt.columns = ['Treat', 'Value']
#       model = ols('Value~C(Treat)', data=df_melt).fit()
#       anova_table = anova_lm(model, typ=2)
#       print(anova_table)
#       mc = MultiComparison(df_melt['Value'], df_melt['Treat'])
#       tukey_result = mc.tukeyhsd(alpha=alpha)
#       print(tukey_result)
#       return tukey_result

#two-way-Anova
# 导入相关库
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#导入数据

d = np.array([
    [276, 352, 178, 295, 273],
    [114, 176, 102, 155, 128],
    [364, 547, 288, 392, 378]
])

df = pd.DataFrame(d)
df.index = pd.Index(['A1', 'A2', 'A3'], name='A')
df.columns = pd.Index(['B1', 'B2', 'B3', 'B4', 'B5'], name='B')


def two_way_ANOVA(df,alpha=0.05):
      """
      aim:研究没有交互作用的双因素方差分析
      :param df: 传入的df是一个dataframe数组，横轴和纵轴分别是两个因素
      :param alpha: 传入置信度
      :return:
      """

      df1 = df.stack().reset_index().rename(columns={0: 'value'})
      model = ols('value~C(A) + C(B)', df1).fit()
      print(anova_lm(model))
      print(pairwise_tukeyhsd(df1['value'], df1['A'], alpha=0.05))  # 第一个必须是销量， 也就是我们的指标


two_way_ANOVA(df)