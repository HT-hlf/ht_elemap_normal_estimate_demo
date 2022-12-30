import cv2
import numpy as np
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud
import math

def PCA(data, sort=True):
    data_mean = np.mean(data, axis=0)
    normalize_data = data - data_mean
    H = np.dot(normalize_data.transpose(), normalize_data)
    eigenvectors, eigenvalues, eigenvectors_transpose = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def normal_image(map):
    map_max = np.max(map)
    map_min = np.min(map)
    map = (map-map_min) /(map_max-map_min)
    return map

image_path='./data/real_stairs_125cm.png'
ele_image = cv2.imread(image_path)
width,height,_=ele_image.shape
local_plane_inclination_threshold=math.cos(45* math.pi/180)
thresholdSquared =5
width,height,_=ele_image.shape

raw_point_cloud_matrix=[]
scale_h= 1
scale_w= 1
line_size=3
for i in range(width):
     for j in range(height):
        raw_point_cloud_matrix.append([i*scale_h,j*scale_w,ele_image[i, j,0] / 255 * 60])
print(ele_image.shape)
raw_point_cloud = DataFrame(raw_point_cloud_matrix)  # 选取每一列的前三个元素[x,y,z]
raw_point_cloud.columns = ['x', 'y', 'z']
point_cloud_pynt = PyntCloud(raw_point_cloud)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
points = point_cloud_pynt.points
print('total points number is:', points.shape[0])
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
o3d.visualization.draw_geometries([point_cloud_o3d, line_set,mesh_frame])
pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
normals = []
squareError_list= []
binaryimage = np.zeros((width,height))

for i in range(points.shape[0]):
    i_h = i % height
    i_w = i // height
    #  eigenvalues,eigenvectors
    if i_h==0 or i_w==0 or i_h==height-1 or i_w==width-1:
        vector = [0, 0, 1]
        squareError = 1e+30
    else:
        k_nearest_point = np.asarray(point_cloud_o3d.points)[
                          [i, i - 1, i + 1, i - height, i - height - 1, i - height + 1, i + height, i + height - 1,
                           i + height + 1], :]

        w, v = PCA(k_nearest_point)
        print('-------------------idex:{}--------------------'.format(i))
        print('点集：',k_nearest_point)
        print('特征值', w)
        print('特征向量', v)
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
        point = [point_cloud_o3d.points[i], point_cloud_o3d.points[i] + v[:, 0] * line_size,
                 point_cloud_o3d.points[i] + v[:, 1] * line_size, point_cloud_o3d.points[i] + v[:, 2] * line_size]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=point_cloud_o3d.points[i])
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

    normals.append(vector)
    if abs(vector[2])>local_plane_inclination_threshold and squareError <thresholdSquared:
        binaryimage[ i_w,i_h]= 1
    else:
        binaryimage[i_w, i_h] = 0

cv2.imshow('binaryimage',binaryimage)
binaryimage=np.uint8(binaryimage)
num_objects, label_image = cv2.connectedComponents(binaryimage,connectivity=8)
print('num_objects:',num_objects)
print(label_image.shape)
label_image_virtual=normal_image(label_image)
label_image_virtual = np.uint8(255 * label_image_virtual)
label_image_virtual = cv2.applyColorMap(label_image_virtual, cv2.COLORMAP_JET)
cv2.imwrite('output.jpg',label_image_virtual)
cv2.imshow('label_image',label_image_virtual)
normals = np.array(normals, dtype=np.float64)
# TODO: 此处把法向量存放在了normals中
point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
o3d.visualization.draw_geometries([point_cloud_o3d,mesh_frame],point_show_normal=True)
cv2.waitKey(0)
cv2.destroyAllWindows()