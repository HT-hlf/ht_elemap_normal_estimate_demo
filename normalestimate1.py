# coding:utf-8
# @Author     : HT
# @Time       : 2022/12/24 10:52
# @File       : normalestimate1.py.py
# @Software   : PyCharm

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:51:41 2022
@author: https://blog.csdn.net/suiyingy
"""

import numpy as np
import open3d as o3d

if __name__ == '__main__':
    file_path = 'bun_zipper.ply'
    ply = o3d.io.read_triangle_mesh(file_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = ply.vertices

    radius = 0.01  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.array(pcd.normals)  # 法向量结果，Nx3

    # 验证法向量模长为1
    res = normals * normals
    res = np.sum(res, axis=1)
    print(res)

    # 法向量可视化
    o3d.visualization.draw_geometries([pcd], window_name="法线估计",
                                      point_show_normal=True,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度

