import numpy as np
import math
import matplotlib.pyplot as plt
import os
eps = 1e-6

def incident_vector_calulate(distance_of_source2object=40,object_size=[20,90],ray_angle=15,ray_size=1,voxels_size=1,attenuation = 0.68):
    '''
    计算入射向量矩阵(每个元素的值代表对应角度，对应体素处的入射向量，向量方向为射线方向，大小为入射强度)
    并返回入射向量矩阵,shape(nray,ny,nz,3) vec = [z,y,x]
    distance_of_source2object: 源到物体前表面的距离
    object_size: 物体的实际尺寸(x=y,z) 单位mm
    ray_angle: 射线角度范围的一半(偏离垂直入射的最大角度) 单位°
    ray_size: 射线角度步长 单位°
    voxels_size: 体素的大小 单位mm
    attenuation: 衰减系数
    '''

    nray = math.floor(ray_angle / ray_size) + 1
    ny = math.floor(object_size[1] / voxels_size) + 1
    nz = math.floor(object_size[0] / voxels_size) + 1
    ny = math.floor(ny * 0.5) + 1
    obj_y,obj_z = object_size
    vector = np.zeros((nray,ny,nz,3))

    rays = np.linspace(0,ray_angle*np.pi/180,nray)

    # voxel_size_y = obj_y / ny;
    # voxel_size_z = obj_z / nz;

    voxel_size_y = voxels_size
    voxel_size_z = voxels_size
    grid_origin =np.array([distance_of_source2object, - 0.5 * voxel_size_y,0])

    
    for p in range(nray):
        ray_vec = np.array([np.cos(rays[p]),np.sin(rays[p]),0])
        d_final = distance_of_source2object + obj_y + obj_z
        final = ray_vec * d_final
        if final[1] < 0.1:
            final[1] = 0.1
        az = np.zeros(nz)
        ay = np.zeros(ny)
        for i in range(nz):
            az[i] = (grid_origin[0]+i*voxel_size_z) / final[0]
        for j in range(ny):
            ay[j] = (grid_origin[1]+j*voxel_size_y) / final[1]
        az_min = min(az[0],az[nz-1])
        az_max = max(az[0],az[nz-1])
        ay_min = min(ay[0],ay[ny-1])
        ay_max = max(ay[0],ay[ny-1])

        a_min = max(az_min,ay_min)
        a_max = min(az_max,ay_max)

        if a_min == az_min:
            i_min = 1
        else:
            i_min = math.ceil((a_min * final[0] - grid_origin[0]) / voxel_size_z)
        if a_max == az_max:
            i_max = nz-1
        else:
            i_max = math.floor((a_max * final[0] - grid_origin[0]) / voxel_size_z)
        if a_min == ay_min:
            j_min = 1
        else:
            j_min = math.ceil((a_min * final[1] - grid_origin[1]) / voxel_size_y)
        if a_max == ay_max:
            j_max = ny-1
        else:
            j_max = math.floor((a_max * final[1] - grid_origin[1]) / voxel_size_y)

        Np = i_max - i_min + j_max - j_min + 2
        a_z = az[i_min]
        a_y = ay[j_min]
        first_i = math.floor((0.5 * (min(a_z,a_y) + a_min) * final[0] - grid_origin[0]) / voxel_size_z)
        first_j = math.floor((0.5 * (min(a_z,a_y) + a_min) * final[1] - grid_origin[1]) / voxel_size_y)
        azu = voxel_size_z / final[0]
        ayu = voxel_size_y / final[1]
        ac = a_min

        i = first_i
        j = first_j
        d12 = d_final * ac
        delta_d = 0 
        intensity = 100
        for q in range(Np+1):
            i12 = intensity * np.exp(-attenuation * delta_d)
            vector[p,j,i,:] = i12 * ray_vec / d12 / d12
            intensity = i12
            if a_z < a_y:
                delta_d = (a_z - ac) * d_final
                d12 = d12 + delta_d
                i = i + 1
                ac = a_z
                a_z = a_z + azu
            else:
                delta_d = (a_y - ac) * d_final
                d12 = d12 + delta_d
                j = j + 1
                ac = a_y
                a_y = a_y + ayu
            if i >= nz or j >= ny:
                break

    # 去掉 y=0 的第一行
    vector_pos = vector[:, 1:, :, :]     # shape (nray, ny-1, nz, 3)

    # 沿 y 维度翻转
    vector_mirror = np.flip(vector_pos, axis=1)  # 翻转 y 方向

    # y 分量取负（索引2对应xyz中的y分量）
    vector_mirror[:, :, :, 1] *= -1

    # 拼接：先镜像部分(y<0)，再原部分(y>=0)
    vector_full = np.concatenate([vector_mirror, vector], axis=1)

        
    return vector_full

def voxel_center_calulate(distance_of_source2object=40,object_size=[20,90],voxels_size=1):
    '''
    计算体素中心坐标并返回体素中心矩阵,shape(ny,nz,3)
    distance_of_source2object: 源到物体前表面的距离
    object_size: 物体的实际尺寸(x=y,z) 单位mm
    voxels_size: 体素的大小 单位mm
    '''
    ny = math.floor(object_size[1] / voxels_size) + 1
    nz = math.floor(object_size[0] / voxels_size) + 1   
    ny = math.floor(ny * 0.5) + 1
    voxel_center = np.zeros((ny,nz,3))

    for i in range(ny):
        for j in range(nz):
            voxel_center[i,j,:] = np.array([distance_of_source2object + i * voxels_size, j * voxels_size,0])       
    # 去掉 y=0 的第一行
    voxel_center_pos = voxel_center[:, 1:, :]     # shape (ny-1, nz, 3)

    # 沿 y 维度翻转
    voxel_center_mirror = np.flip(voxel_center_pos, axis=0)  # 翻转 y 方向

    # y 分量取负（索引2对应xyz中的y分量）
    voxel_center_mirror[:, :, 1] *= -1

    # 拼接：先镜像部分(y<0)，再原部分(y>=0)
    voxel_center_full = np.concatenate([voxel_center_mirror, voxel_center], axis=0)
    return voxel_center_full

def voxel_path_length_cal(grid_origin,grid_size,nx,ny,nz,ray_vec,ray_origin,attenuation = 0.68):
    '''
    计算体素路径长度
    grid_origin: 体素网格的原点坐标 [z,y,x]
    grid_size: 体素的大小
    nx,ny,nz: 体素网格在x,y,z方向的数量
    ray_vec: 入射向量组,shape(num,3) 模为射线强度 vec = [z,y,x]
    ray_origin: 入射向量起始点数组,输入为网格序号,shape(num,3) vec = [z,y,x]
    attenuation: 衰减系数
    '''
    # ptr_in = m_ptr
    # data_in = data
    # n_idx_in = idx
    # num = len(ptr_in)+1
    # m_idx_in = np.zeros(num)

    # for i in range(num):
    #     m_idx_in[ptr_in[i]:ptr_in[i+1]-1] = i

    # for i in range(num):
    #     vec = data_in[i]
    #     vec_len = np.linalg.norm(vec)
    #     vec_origin = grid_origin + np.array([n_idx_in[i] * grid_size[0],n_idx_in[i] * grid_size[1],0])
    #     vec_end = vec_origin + vec / vec_len * ()

    num = np.shape(ray_vec)[0]
    # 用list动态存储
    vec_out = []
    nums, zs, ys, xs = [], [], [], []

    for num_i in range(num):
        soc = grid_origin + grid_size * (ray_origin[num_i]+0.5)
        vec_intensity = np.linalg.norm(ray_vec)
        vec = ray_vec[num_i] / vec_intensity
        d = nx + ny + nz
        final_vec = soc + d * vec
        grid_x = grid_size
        grid_y = grid_size 
        grid_z = grid_size
        az = np.zeros(nz)
        ay = np.zeros(ny)
        ax = np.zeros(nx)
        for i in range(nz):
            az[i] = (grid_origin[0]+i*grid_z-soc[0]) / (final_vec[0] - ray_origin[num_i,0] + eps)
        for j in range(ny):
            ay[j] = (grid_origin[1]+j*grid_y-soc[1]) / (final_vec[1] - ray_origin[num_i,1] + eps)
        for k in range(nx):
            ax[k] = (grid_origin[2]+k*grid_x-soc[2]) / (final_vec[2] - ray_origin[num_i,2] + eps)

        az_min = min(az[0],az[nz-1])
        az_max = max(az[0],az[nz-1])
        ay_min = min(ay[0],ay[ny-1])
        ay_max = max(ay[0],ay[ny-1])
        ax_min = min(ax[0],ax[nx-1])
        ax_max = max(ax[0],ax[nx-1])

        a_min = max(az_min,ay_min,ax_min)
        a_max = min(az_max,ay_max,ax_max)

        if soc[0]<final_vec[0]:
            if a_min == az_min:
                i_min = 1
            else:
                i_min = math.ceil((soc[0] + a_min * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            if a_max == az_max:
                i_max = nz-1
            else:
                i_max = math.floor((soc[0] + a_max * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            a_z = az[i_min]
            iu = 1
        else:
            if a_min == az_min:
                i_max = nz-2
            else:
                i_max = math.floor((soc[0] + a_min * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            if a_max == az_max:
                i_min = 0
            else:
                i_min = math.ceil((soc[0] + a_max * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            a_z = az[i_max]
            iu = -1

        if soc[1]<final_vec[1]:
            if a_min == ay_min:
                j_min = 1
            else:
                j_min = math.ceil((soc[1] + a_min * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
            if a_max == ay_max:
                j_max = ny-1
            else:
                j_max = math.floor((soc[1] + a_max * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
            a_y = ay[j_min]
            ju = 1
        else:
            if a_min == ay_min:
                j_max = ny-2
            else:
                j_max = math.floor((soc[1] + a_min * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
            if a_max == ay_max:
                j_min = 0
            else:
                j_min = math.ceil((soc[1] + a_max * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
            a_y = ay[j_max]
            ju = -1

        if soc[2]<final_vec[2]:
            if a_min == ax_min:
                k_min = 1
            else:
                k_min = math.ceil((soc[2] + a_min * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
            if a_max == ax_max:
                k_max = nx-1
            else:
                k_max = math.floor((soc[2] + a_max * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
            a_x = ax[k_min]
            ku = 1
        else:
            if a_min == ax_min:
                k_max = nx-2
            else:
                k_max = math.floor((soc[2] + a_min * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
            if a_max == ax_max:
                k_min = 0
            else:
                k_min = math.ceil((soc[2] + a_max * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
            a_x = ax[k_max]
            ku = -1

        Np = i_max - i_min + j_max - j_min + k_max - k_min + 3

        first_i = math.floor((soc[0]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
        first_j = math.floor((soc[1]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
        first_k = math.floor((soc[2]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
        azu = iu * grid_z / (final_vec[0]-soc[0])
        ayu = ju * grid_y / (final_vec[1]-soc[1])
        axu = ku * grid_x / (final_vec[2]-soc[2])
        ac = a_min

        i = first_i
        j = first_j
        k = first_k
        d12 = 0
        if ac >0:
            d12 = d * ac
        delta_d = 0 
        intensity = vec_intensity
        # for q in range(Np+1):
        while True:
            i12 = intensity * np.exp(-attenuation * delta_d)
            if d12 > 0:
                nums.append(num_i)
                zs.append(i)
                ys.append(j)
                xs.append(k)
                vec_out.append(i12 * vec / d12 / d12)
            intensity = i12
            if a_z < a_y and a_z < a_x:
                if ac >0:
                    delta_d = (a_z - ac) * d
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                i = i + iu
                ac = a_z
                a_z = a_z + azu
            elif a_y < a_x and a_y < a_z:
                if ac >0:
                    delta_d = (a_y - ac) * d
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                j = j + ju
                ac = a_y
                a_y = a_y + ayu
            else:
                if ac >0:
                    delta_d = (a_x - ac) * d
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                k = k + ku
                ac = a_x
                a_x = a_x + axu
            if i >= nz or i < 0 or j >= ny or j < 0 or k >= nx or k < 0:
                break


    print("total voxels:",len(xs))
    return {
            # "m_ptr": ptr_out,
            # "data": data_out.astype(np.float32, copy=False),
            # "idx": n_idx_out.astype(np.int32, copy=False),
            # "shape": shape_out
            "num": np.array(nums).astype(np.int32, copy=False),
            "z": np.array(zs).astype(np.int32, copy=False),
            "x": np.array(xs).astype(np.int32, copy=False),
            "y": np.array(ys).astype(np.int32, copy=False),
            "vec": np.array(vec_out).astype(np.float32, copy=False)
        }