import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from incident import incident_vector_calulate,voxel_center_calulate,voxel_path_length_cal
from visualize_ray import visualize_ray
# # 假设
# nray = 5
# ny, nz = 40, 180
# angle = 15*np.pi/180
# rays = np.linspace(0, np.pi, nray)

# # 假设你的函数生成了 vector
# vector = incident_vector_calulate(distance_of_source2object=40,object_size=[20,90],ray_angle=angle,n_voxels_shape=[nray,ny,nz],attenuation = 0.28)

# # 可视化第 3 个射线角度
# for i in range(5):
#     visualize_ray(vector, angle_index=i)

# 示例参数（你要改成实际的）
grid_origin = np.array([0.0, 0.0, 0.0])
grid_size =  1.0
nx, ny, nz = 10, 10, 10

ray_vec = np.array([[-1, 2, -3]])   # 只射一条线，方向(1,2,3)
ray_origin = np.array([[5, 4, 6]])  # 从原点出发

result = voxel_path_length_cal(
    grid_origin, grid_size,
    nx, ny, nz,
    ray_vec, ray_origin
)

# ==== 3D 可视化 ====
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 提取非零体素坐标
xs = result["x"]
ys = result["y"]
zs = result["z"]

ax.scatter(xs, ys, zs, c='red', marker='o', s=40, label="nonzero voxels")

# 设置坐标范围
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_zlim(0, nz)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()