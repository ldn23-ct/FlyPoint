import numpy as np
from incident import incident_vector_calulate
from visualize_ray import visualize_ray
# 假设
nray = 5
ny, nz = 40, 180
angle = 15*np.pi/180
rays = np.linspace(0, np.pi, nray)

# 假设你的函数生成了 vector
vector = incident_vector_calulate(distance_of_source2object=40,object_size=[20,90],ray_angle=angle,n_voxels_shape=[nray,ny,nz],attenuation = 0.28)

# 可视化第 3 个射线角度
for i in range(5):
    visualize_ray(vector, angle_index=i)
