import matplotlib.pyplot as plt
import numpy as np

if RA_flag==1:
    MVC_10 = gesture_oversub_score[:,0,:]
    MVC_40 = gesture_oversub_score[:,1,:]
    MVC_70 = gesture_oversub_score[:,2,:]
    MVC_all = gesture_oversub_score[:,3,:]

    print(1)
    feature_set_cata = [MVC_10, MVC_40, MVC_70, MVC_all]
    labels_name = ['RF', 'LDA', 'KNN', 'SVM']
    MVC_level = ['MVC_10', 'MVC_40', 'MVC_70', 'MVC_all']

    color_list = ['skyblue', 'lightcoral', 'navajowhite', 'limegreen']

    for i in range(4):
        x = np.arange(4)
        y = np.mean(feature_set_cata[i], axis=0)
        error = np.std(feature_set_cata[i], axis=0)
        plt.bar(x + 0.2 * i, y * 100, alpha=0.6, width=0.2, lw=3, label=MVC_level[i], color=color_list[i])
        plt.errorbar(x + 0.2 * i, y * 100, yerr=error * 100, fmt='.', ecolor='black',
                     elinewidth=1, ms=5, mfc='wheat', mec='salmon', capsize=5)
    plt.xticks(x + 0.3, labels_name, fontsize=12)
    plt.xlabel('Machine learning Model', loc='center', fontsize=12, weight='medium')
    plt.ylabel('Recognition Accuracy (%) ', fontsize=12)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('C:\\Users\\86156\\Desktop\\f31.png')
    plt.show()