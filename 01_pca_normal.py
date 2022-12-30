#!/home/ycz/anaconda3/envs/cloud_lesson/bin/python
# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
import os
import random
import linecache
import numpy as np
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud


def get_file_path():
    root_dir = 'Z:\Doc\normalestimate\txt'  # 数据集文件夹
    data_list = 'bed_0003.txt'  # 数据集目录
    data_path = root_dir + data_list
    getlines = linecache.getlines(data_path)

    dir_rdn = random.randint(1, len(getlines))
    file_dir = linecache.getline(data_path, dir_rdn)
    file_path = os.listdir(root_dir + file_dir[:-1])
    txt_rdn = random.randint(1, len(file_path) - 1)
    filename = root_dir + file_dir[:-1] + "/" + file_path[txt_rdn]
    return filename


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#    correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    # 对列求均值 data => (10000, 3)  data_mean => (1, 3)
    data_mean = np.mean(data, axis=0)
    # 数据归一化操作 normalize_data => (10000, 3)
    normalize_data = data - data_mean
    # H => (3, 3)
    H = np.dot(normalize_data.transpose(), normalize_data)
    # SVD分解eigenvectors => (3,3)  eigenvalues => (3,)  eigenvectors_transpose => (3,3)
    eigenvectors, eigenvalues, eigenvectors_transpose = np.linalg.svd(H)
    # 屏蔽结束
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def main():
    # 1.加载自己的点云文件
    # filename = get_file_path()
    # raw_point_cloud_matrix => (10000, 6)
    # raw_point_cloud_matrix = np.genfromtxt(filename, delimiter=",")
    raw_point_cloud_matrix = np.genfromtxt("txt/bed_0003.txt", delimiter=",")
    raw_point_cloud = DataFrame(raw_point_cloud_matrix[:, :3])  # 选取每一列的前三个元素[x,y,z]
    raw_point_cloud.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(raw_point_cloud)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])
    # 2.用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # 3.构造open3d中的LineSet对象，用于主成分和次主成分显示
    point = [[0, 0, 0], v[:, 0], v[:, 1],v[:, 2]]
    lines = [[0, 1], [0, 2],[0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])
    # 4.循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    # 屏蔽开始
    # 每一点的法向量计算，通过PCA降维，对应最小特征值的成分向量近似为法向量
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 20)
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])
    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
