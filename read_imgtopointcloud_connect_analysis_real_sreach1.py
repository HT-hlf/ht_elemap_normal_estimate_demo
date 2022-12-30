# coding:utf-8
# @Author     : HT
# @Time       : 2022/12/24 8:08
# @File       : read_imgtopointcloud.py.py
# @Software   : PyCharm

import cv2

import os
import random
import linecache
import numpy as np
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud
import math


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
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
    # print(eigenvalues)
    return eigenvalues, eigenvectors

def normal_image(map):


    map_max = np.max(map)
    map_min = np.min(map)

    # heatmap = np.maximum(heatmap, 0)
    map = (map-map_min) /(map_max-map_min)

    return map

# image_path=r'Y:\tra_gridmap_img_matrix_class/ht.jpg'
# image_path='./data/real_stairs_125cm.png'
# image_path='./data/terrain.png'
image_path='./data/custom1.png'
ele_image = cv2.imread(image_path)
width,height,_=ele_image.shape
# ele_image = cv2.resize(ele_image ,(height*4,width*4))
local_plane_inclination_threshold=math.cos(45* math.pi/180)
thresholdSquared =0.5
# cv2.imshow('terrain',ele_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
width,height,_=ele_image.shape

raw_point_cloud_matrix=[]
scale_h= 1
scale_w= 1
line_size=3
for i in range(width):
     for j in range(height):
         # if ele_image[i, j,0]<25:
        raw_point_cloud_matrix.append([i*scale_h,j*scale_w,ele_image[i, j,0] / 255 * 60])
print(ele_image.shape)
# print(ele_image[20,20,:])

raw_point_cloud = DataFrame(raw_point_cloud_matrix)  # 选取每一列的前三个元素[x,y,z]
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
point = [[0, 0, 0], v[:, 0], v[:, 1]]
lines = [[0, 1], [0, 2]]
colors = [[1, 0, 0], [0, 1, 0]]
line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
line_set.colors = o3d.utility.Vector3dVector(colors)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([point_cloud_o3d, line_set,mesh_frame]
#                                   # ,width=1200,height=1000
#                                    )
# 4.循环计算每个点的法向量
pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
normals = []
squareError_list= []
# binarylist = []
binaryimage = np.zeros((width,height))
# 作业2
# 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
# 屏蔽开始
# 每一点的法向量计算，通过PCA降维，对应最小特征值的成分向量近似为法向量
for i in range(points.shape[0]):
    # [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 40)
    # print('idx:',idx)
    i_h = i % height
    i_w = i // height
    #  eigenvalues,eigenvectors
    if i_h==0 or i_w==0 or i_h==height-1 or i_w==width-1:
        vector = [0, 0, 1]
        squareError = 1e+30
    else:
        # print(i)
        # print(i_h,i_w)
        # print(width,height)
        k_nearest_point = np.asarray(point_cloud_o3d.points)[
                          [i, i - 1, i + 1, i - height, i - height - 1, i - height + 1, i + height, i + height - 1,
                           i + height + 1], :]

        w, v = PCA(k_nearest_point)
        # print('-------------------idex:{}--------------------'.format(i))
        # print('点集：',k_nearest_point)
        # print('特征值', w)
        # print('特征向量', v)
        if w[1]>1e-8:
            vector=v[:, 2]
            if vector[2]<0:
                vector=-vector
            squareError= w[2] if (w[2]>0)  else 0
        else:
            vector = [0,0,1]
            squareError = 1e+30
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(k_nearest_point)
        # o3d.visualization.draw_geometries([pcd])
        # vector = np.array([vector], dtype=np.float64)
        # TODO: 此处把法向量存放在了normals中
        # pcd.normals = o3d.utility.Vector3dVector(vector)
        # o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

        # point = [[0, 0, 0], v[:, 0], v[:, 1],v[:, 2]]
        point = [point_cloud_o3d.points[i], point_cloud_o3d.points[i] + v[:, 0] * line_size,
                 point_cloud_o3d.points[i] + v[:, 1] * line_size, point_cloud_o3d.points[i] + v[:, 2] * line_size]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=point_cloud_o3d.points[i])
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # save_view_point(pcd, "viewpoint.json")
        # load_view_point(pcd, "viewpoint.json")

        # if i==7146:
        # if 100>w[2] >5 and i>4000:
        #     o3d.visualization.draw_geometries([pcd,line_set,mesh_frame],width=1400,height=1000,
        #                                       # front=[0.5, 0.86, 0.125],
        #                                       # lookat=[0.23, 0.5, 2],
        #                                       # up=[-0.63, 0.45, -0.63],
        #                                       # zoom=0.7
        #                                       )




    # print(squareError)
    normals.append(vector)
    # squareError_list.append(squareError)
    # print([i_h,i_w])

    if abs(vector[2])>local_plane_inclination_threshold and squareError <thresholdSquared:
        # binarylist.append(True)
        binaryimage[ i_w,i_h]= 1
    else:
        # binarylist.append(False)
        binaryimage[i_w, i_h] = 0



# k = np.ones((10, 10), np.uint8)
# binaryimage=cv2.morphologyEx(binaryimage, cv2.MORPH_OPEN,k)
cv2.imshow('binaryimage',binaryimage)
binaryimage=np.uint8(binaryimage)
num_objects, label_image = cv2.connectedComponents(binaryimage,connectivity=8)
print('num_objects:',num_objects)
print(label_image.shape)
label_image_virtual=normal_image(label_image)
label_image_virtual = np.uint8(255 * label_image_virtual)  # 将热力图转换为RGB格式
label_image_virtual = cv2.applyColorMap(label_image_virtual, cv2.COLORMAP_JET)
# label_image_virtual = cv2.resize(label_image_virtual ,(height*4,width*4))
cv2.imwrite('output.jpg',label_image_virtual)
cv2.imshow('label_image',label_image_virtual)



# 屏蔽结束
normals = np.array(normals, dtype=np.float64)
# TODO: 此处把法向量存放在了normals中
point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([point_cloud_o3d, line_set])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
o3d.visualization.draw_geometries([point_cloud_o3d,mesh_frame],point_show_normal=True)
cv2.waitKey(0)
cv2.destroyAllWindows()