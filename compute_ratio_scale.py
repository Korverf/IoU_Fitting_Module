import os
import json
import numpy as np
#from sklearn.cluster import KMeans
from matplotlib import pyplot
from matplotlib.ticker import PercentFormatter
import pandas as pd

CLASSES = ['A','B','C','D','E','F','G','H','I','J','K']

#def annotation_convert_5to8():

def annotation_convert_8to5(annotations):
    preds = np.array(annotations, dtype=np.float32)
    preds = np.reshape(preds, newshape=(-1,2,4),order='F')
    angle = np.arctan2(-(preds[:,0,1]-preds[:,0,0]), preds[:,1,1]-preds[:,1,0])+np.pi
    center = np.zeros((preds.shape[0],2,1))
    for i in range(4):
        center[:,0,0]+=preds[:,0,i]
        center[:,1,0]+=preds[:,1,i]
    center = np.array(center, dtype=np.float32)/4.0

    R=np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    normalized = np.matmul(R.transpose((2,1,0)), preds-center)

    xmin = np.min(normalized[:,0,:], axis=1)
    xmax = np.max(normalized[:,0,:], axis=1)
    ymin = np.min(normalized[:,1,:], axis=1)
    ymax = np.max(normalized[:,1,:], axis=1)

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    w = w[:,np.newaxis]
    h = h[:,np.newaxis]

    angle = angle[:,np.newaxis]

    return np.concatenate((center[:,0], center[:,1], w, h, angle), axis=1)

def generate_axes(lists, interval, max_value):
    index = list(np.arange(interval/2, max_value+interval/2, interval))
    cut = pd.cut(lists, index)
    counts = pd.value_counts(cut).sort_index()

    x = np.arange(interval, max_value, interval)
    x = np.around(x, 1)
    x = [str(i) for i in x]
    y = counts.values

    return x, y
    
def get_ratio_scale(category_id, annotations, ratio_interval, scale_interval):
    ratio_lists=[]
    scale_lists=[]

    for item in annotations:
        ratio_lists.append(item[2]/item[3])
        scale_lists.append(item[2]*item[3])

    fig1, axs = pyplot.subplots(2, figsize=(10,7))
    if category_id != None:
        fig1.suptitle('Class: ' + category_id, fontsize=16)
    else:
        fig1.suptitle('All classes', fontsize=16)

    # ratio
    x, y = generate_axes(ratio_lists, ratio_interval, 4)
    y = [item/len(annotations) for item in y]

    axs[0].bar(x,y)
    axs[0].set_title('ratios(interval= %.1f )' % ratio_interval, fontsize=15)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1)) # 纵轴显示百分数

    # scale
    x, y = generate_axes(scale_lists, scale_interval, 40000)
    y = [item/len(annotations) for item in y]

    axs[1].bar(x,y)
    axs[1].set_title('scales(interval= %d )' % scale_interval, fontsize=15)
    axs[1].yaxis.set_major_formatter(PercentFormatter(1)) # 纵轴显示百分数

    # pyplot.show()
    return

def get_ratio_scale_kmeans(annotations):
    data = np.empty((len(annotations),4))
    for idx, item in enumerate(annotations):
        data[idx,0] = item[2]/item[3]
        data[idx,1] = 0
        data[idx,2] = item[2]*item[3]
        data[idx,3] = 0
    pyplot.scatter(data[:,2], data[:,3], s=1)
    pyplot.show()
    # clf = KMeans(n_clusters=5)
    # clf.fit(data[:,2:])
    #
    # centers = clf.cluster_centers_
    # print(centers)
    # labels = clf.labels_
    #
    # for i in range(len(labels)):
    #     pyplot.scatter(data[i,0],data[i][1],c=('r' if labels[i]==0 else 'b'))
    # pyplot.scatter(centers[:,0], centers[:,1], marker='*', s=100)
    # pyplot.show()
    return 0,0

def read_annotation(file):
    imgs_list = []
    annotations_list = {cat_:[] for cat_ in CLASSES}
    for item in file:
        image_name = item['image_name']
        labels = item['labels']

        imgs_list.append(image_name)
        for label in labels:
            category_id = label['category_id']
            points = label['points']
            annotations_list[category_id].append(np.array([points[0][0],points[0][1],points[1][0],points[1][1],
                                                  points[2][0],points[2][1],points[3][0],points[3][1]]))

    return imgs_list, annotations_list

def compute_ratio_scale(annotations_list, ratio_interval, scale_interval):
    categories = annotations_list.keys()
    annotations_all = []
    nums_per_cat = []

    for cat_ in categories:
        annotations = annotations_list[cat_]    # 每个类别下的所有数据
        nums_per_cat.append(len(annotations))
        annotations = annotation_convert_8to5(annotations)

        get_ratio_scale(cat_, annotations, ratio_interval, scale_interval)
        annotations_all.append(annotations)
        # ratio, ratio_max, ratio_min, scale, scale_max, scale_min = get_ratio_scale(annotations)
        # print(cat_,': ',ratio, ' ',ratio_max,' ',ratio_min, '| ',scale, ' ', scale_max, ' ', scale_min)
    annotations_all = np.concatenate(annotations_all, axis=0)

    get_ratio_scale(None, annotations_all, ratio_interval, scale_interval)
    count_categories(annotations_all, nums_per_cat)
    return

def count_categories(annotations, nums_per_cat):
    x = CLASSES
    y = nums_per_cat
    y = [item/len(annotations) for item in y]

    fig1, ax1 = pyplot.subplots(figsize=(10,7))
    ax1.bar(x,y)
    ax1.set_title('catagories', fontsize=15)
    pyplot.gca().yaxis.set_major_formatter(PercentFormatter(1)) # 纵轴显示百分数

    # pyplot.show()
    return


if __name__=='__main__':
    f=open('/home/yyw/yyf/Dataset/rsaicp/train_all.json','r')
    f=json.load(f)
    imgs_list, annotations_list = read_annotation(f)

    ratio_interval = 0.3    # 调整ratio间隔
    scale_interval = 5000   # 调整scale间隔
    compute_ratio_scale(annotations_list, ratio_interval, scale_interval)

    pyplot.show()