#!/home/ycz/anaconda3/envs/cloud_lesson/bin/python
# 实现voxel滤波，并加载数据集中的文件进行验证
import os
import random
import linecache
import numpy as np
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud


def get_file_path():
    root_dir = '/home/ycz/ws/py/modelnet40_normal_resampled/'  # 数据集文件夹
    data_list = 'modelnet40_shape_names.txt'  # 数据集目录
    data_path = root_dir + data_list
    getlines = linecache.getlines(data_path)
    dir_rdn = random.randint(1, len(getlines))
    file_dir = linecache.getline(data_path, dir_rdn)
    file_path = os.listdir(root_dir + file_dir[:-1])
    txt_rdn = random.randint(1, len(file_path) - 1)
    filename = root_dir + file_dir[:-1] + "/" + file_path[txt_rdn]
    return filename


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, mode):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # step1: Compute the min / max of each coordinate
    x_max, y_max, z_max = point_cloud.max(axis=0)
    x_min, y_min, z_min = point_cloud.min(axis=0)
    print(x_max, y_max, z_max)
    print(x_min, y_min, z_min)
    # step2: Compute the dimension of the voxel grid
    Dx = (x_max - x_min) / leaf_size
    Dy = (y_max - y_min) / leaf_size
    Dz = (z_max - z_min) / leaf_size
    # step3: Compute voxel index for each point
    point_x = np.array(point_cloud.x)
    point_y = np.array(point_cloud.y)
    point_z = np.array(point_cloud.z)
    hx = np.floor((point_x - x_min) / leaf_size)
    hy = np.floor((point_y - y_min) / leaf_size)
    hz = np.floor((point_z - z_min) / leaf_size)
    H = np.array(np.floor(hx + hy * Dx + hz * Dx * Dy))
    # step4: Sort the points according to the index in step4
    data = np.c_[H, point_x, point_y, point_z]
    data = data[data[:, 0].argsort()]
    # step5: Iterate the sorted points, select points according to Centroid / Random method
    # 随机采样
    if mode == "random":
        filtered_points = []
        for i in range(data.shape[0] - 1):
            # 判断h是否相等
            if (data[i][0] != data[i + 1][0]):
                # 选取序号为h的栅格内第1点作为随机采样点
                filtered_points.append(data[i][1:])
            filtered_points.append(data[data.shape[0] - 1][1:])
    # 均值采样
    if mode == "centroid":
        filtered_points = []
        data_points = []
        for i in range(data.shape[0] - 1):
            # 判断h是否相等
            if (data[i][0] != data[i + 1][0]):
                data_points.append(data[i][1:])
                continue
            if data_points == []:
                continue
                # 选取序号为h的栅格内第1点作为随机采样点
            filtered_points.append(np.mean(data_points, axis=0))
            data_points = []
        filtered_points=np.array(filtered_points)
    # 屏蔽结束
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # 1.加载自己的点云文件
    # filename = get_file_path()
    # raw_point_cloud_matrix => (10000, 6)
    # raw_point_cloud_matrix = np.genfromtxt(filename, delimiter=",")
    raw_point_cloud_matrix = np.genfromtxt("txt/car_0007.txt", delimiter=",")
    raw_point_cloud = DataFrame(raw_point_cloud_matrix[:, :3])  # 选取每一列的前三个元素[x,y,z]
    raw_point_cloud.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(raw_point_cloud)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    # 调用voxel滤波函数，实现滤波 centroid/random
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.07, "centroid")
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
