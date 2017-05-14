import numpy as np
from operator import itemgetter


def knn(inX, dataSet, labels, k):
    '''
    :param inX: 需预测的点，一维数组表示
    :param dataSet: 给定数据集，二维数组，每行代表一条数据
    :param labels: 数据集中每条数据对应的标签列表
    :return: 返回距离最近的标签
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSet,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
