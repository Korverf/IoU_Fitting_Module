import os
import json
import numpy as np
#from sklearn.cluster import KMeans
from matplotlib import pyplot
from matplotlib.ticker import PercentFormatter
import pandas as pd


def generate_axes(lists, interval, max_value):
    index = list(np.arange(interval/2, max_value+interval/2, interval))
    cut = pd.cut(lists, index)
    counts = pd.value_counts(cut).sort_index()

    x = np.arange(interval, max_value, interval)
    x = np.around(x, 1)
    x = [str(i) for i in x]
    y = counts.values

    return x, y

if __name__=='__main__':


    ratio_interval = 0.5    # 调整ratio间隔
    scale_interval = 5000   # 调整scale间隔
    width_interval = 25  # 调整width间隔
    height_interval = 25    # 调整height间隔

    # ratio
    x, y = generate_axes(ratio_lists, ratio_interval, 10)
    y = [item / len(annotations) for item in y]

    axs[0].bar(x, y)
    axs[0].set_title('ratios(interval= %.1f )' % ratio_interval, fontsize=15)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))  # 纵轴显示百分数

    pyplot.show()