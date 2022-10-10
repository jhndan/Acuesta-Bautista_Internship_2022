from turtle import down
import numpy as np 
import open3d as o3d

pcd = o3d.io.read_point_cloud("sample.pcd" )
print (pcd)
print (np.asarray(pcd.points))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
#Downpcd
# downpcd = pcd.voxel_down_sample()
# print (downpcd)

#color
# pcd.paint_uniform_color ([1,0.706,0])


#HYBRID
pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1 , max_nn = 30))
o3d.visualization.draw_geometries([pcd])




# o3d.visualization.draw_geometries([downpcd])

