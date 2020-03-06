from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#加载数据,load/save针对二进制文件，loadtxt/savetxt/genfromtxt针对文本文件,一行是一个[一维数组]
def loadDataSet(fileName):
    data = np.genfromtxt(fileName,delimiter=' ')
    return data

#计算距离
def calculate_Distance(vector1,vector2):
    return np.sqrt(sum((vector2 - vector1) ** 2))

#初始化质心
def initCenter(data,k):
    numSamples,dim = data.shape
    # k个质心,列数跟样本的列数一样
    center = np.zeros((k,dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引,np.random.uniform(low,high,size)产生在[0,1]中均匀分布的随机数
        index = int(np.random.uniform(0, numSamples))
        # 作为初始化的质心
        center[i,:] = data[index,:]
    return center

#kmeans实现
#传入数据集和k值
def kmeans(data, k):
    # 计算样本个数
    numSamples = data.shape[0]
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clusterData = np.array(np.zeros((numSamples, 2)))
    # 决定质心是否要改变的质量
    clusterChanged = True
    # 初始化质心
    centroids = initCenter(data, k)
    while clusterChanged:
        clusterChanged = False
        # 循环每一个样本
        for i in range(numSamples):
            # 最小距离
            minDist = 100000.0
            # 定义样本所属的簇
            minIndex = 0
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                # 循环每一个质心和样本，计算距离
                distance = calculate_Distance(centroids[j, :], data[i, :])
                # 如果计算的距离小于最小距离，则更新最小距离
                if distance < minDist:
                    minDist = distance
                    # 更新最小距离
                    clusterData[i, 1] = minDist
                    # 更新样本所属的簇
                    minIndex = j
            # 如果样本的所属的簇发生了变化
            if clusterData[i, 0] != minIndex:
                # 质心要重新计算
                clusterChanged = True
                # 更新样本的簇
                clusterData[i, 0] = minIndex
        # 更新质心
        for j in range(k):
            # 获取第j个簇所有的样本所在的索引
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            # 第j个簇所有的样本点
            pointsInCluster = data[cluster_index]
            # 计算质心
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
    return centroids, clusterData

#结果可视化
def showCluster(data, k, centroids, clusterData):
    numSamples, dim = data.shape
    if dim != 2:
        print('dimension of your data is not 2!')
        return 1
    # 用不同颜色形状来表示各个类别
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'dr', '<r', 'pr']
    if k>len(mark):
        print('your k is too large!')
        return 1
    # 画样本点
    for i in range(numSamples):
        markIndex = int(clusterData[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    # 用不同颜色形状来表示各个类别
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=20)
    plt.show()


if __name__=='__main__':
    data=loadDataSet('kmeans.txt')
    # data=loadDataSet(r'E:\Anaconda3\Lib\site-packages\sklearn\datasets\data\iris.csv')
    center=initCenter(data,3)
    centroids, clusterData=kmeans(data,3)
    showCluster(data,3,centroids,clusterData)




