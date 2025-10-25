import numpy as np
import math
import matplotlib.pyplot as plt
import os
eps = 1e-6

def incident_vector_calulate(distance_of_source2object=40,object_size=[200,90],ny=200,nz=90,ray_angle=15*np.pi/180,ray_step=1*np.pi/180,voxels_size=[1,1,1],miu=np.ones((200,90))):
    '''
    计算入射向量矩阵(每个元素的值代表对应角度，对应体素处的入射向量，向量方向为射线方向，大小为入射强度)
    input:
        distance_of_source2object: 源到物体前表面的距离
        object_size: 物体的实际尺寸(x,y,z) 单位mm
        ny,nz: y,z方向网格数量,要求ny为偶数
        ray_angle: 射线角度范围的一半(偏离垂直入射的最大角度) 单位°
        ray_step: 射线角度步长 单位°
        voxels_size: 体素的大小[z,y,x] 单位mm
        miu: 衰减系数[ny,nz] 单位 1/mm
    output:
        {
            "m_ptr": m_ptr,shape(ny*nz+1),体素指针 ny=j,nz=i 对应 m=nz*j+i
            "p_idx": p_idx,shape(num_nonzero),非零体素在vector中的索引
            "vec": np.array(vec_out_total).astype(np.float32, copy=False)，出射方向单位向量
            "data": np.array(data_out).astype(np.float32, copy=False)，出射向量强度
    '''

    nray = math.floor(ray_angle / ray_step) + 1
    # nray = math.floor(nray * 0.5) + 1
    # ny = math.floor(object_size[1] / voxels_size[0]) + 1
    # nz = math.floor(object_size[0] / voxels_size) + 1
    # ny = math.floor(ny * 0.5) + 1
    # ny = math.floor(ny * 0.5)
    obj_y,obj_z = object_size
    data = np.zeros((nray,ny,nz))

    rays = np.linspace(-ray_angle/2,ray_angle/2,nray)

    # voxel_size_y = obj_y / ny;
    # voxel_size_z = obj_z / nz;

    voxel_size_y = voxels_size[1]
    voxel_size_z = voxels_size[2]
    # grid_origin =np.array([distance_of_source2object, - 0.5 * voxel_size_y,0])
    grid_origin = np.array([distance_of_source2object, -0.5 * obj_y , 0])
    critical_angle = math.atan(0.5 * obj_y / distance_of_source2object)
    for p in range(len(rays)):
        if math.fabs(rays[p]) > critical_angle:
            break
        ray_vec = np.array([np.cos(rays[p]),np.sin(rays[p]),0])
        d_final = distance_of_source2object + 0.5 * obj_y + obj_z
        final = ray_vec * d_final
        if math.fabs(final[1]) < 0.1:
            final[1] = 0.1
        az = np.zeros(nz+1)
        ay = np.zeros(ny+1)
        for i in range(nz+1):
            az[i] = (grid_origin[0]+i*voxel_size_z) / final[0]
        for j in range(ny+1):
            ay[j] = (grid_origin[1]+j*voxel_size_y) / final[1]
        az_min = min(az[0],az[nz])
        az_max = max(az[0],az[nz])
        ay_min = min(ay[0],ay[ny])
        ay_max = max(ay[0],ay[ny])

        a_min = max(az_min,ay_min)
        a_max = min(az_max,ay_max)
        if a_min == az_min:
            i_min = 1
        else:
            i_min = math.ceil((a_min * final[0] - grid_origin[0]) / voxel_size_z)
        if a_max == az_max:
            i_max = nz
        else:
            i_max = math.floor((a_max * final[0] - grid_origin[0]) / voxel_size_z)
        a_z = az[i_min]
        iu = 1
        if ray_vec[1] >= 0:
            if a_min == ay_min:
                j_min = 1
            else:
                j_min = math.ceil((a_min * final[1] - grid_origin[1]) / voxel_size_y)
            if a_max == ay_max:
                j_max = ny
            else:
                j_max = math.floor((a_max * final[1] - grid_origin[1]) / voxel_size_y)
            a_y = ay[j_min]
            ju = 1
        else:
            if a_min == ay_min:
                j_max = ny-1
            else:
                j_max = math.floor((a_min * final[1] - grid_origin[1]) / voxel_size_y)
            if a_max == ay_max:
                j_min = 0
            else:
                j_min = math.ceil((a_max * final[1] - grid_origin[1]) / voxel_size_y)
            a_y = ay[j_max]
            ju = -1
        Np = i_max - i_min + j_max - j_min + 2

        first_i = math.floor((0.5 * (min(a_z,a_y) + a_min) * final[0] - grid_origin[0]) / voxel_size_z)
        first_j = math.floor((0.5 * (min(a_z,a_y) + a_min) * final[1] - grid_origin[1]) / voxel_size_y)
        azu = iu * voxel_size_z / final[0]
        ayu = ju * voxel_size_y / final[1]
        ac = a_min

        i = first_i
        j = first_j
        d12 = 0
        delta_d = 0 
        intensity = 1
        for q in range(Np+1):
            # i12 = intensity * np.exp(-attenuation * d12)
            # data[p,j,i] = i12
            if a_z < a_y:
                i12 = intensity * np.exp(-d12)
                data[p,j,i] = i12
                delta_d = (a_z - ac) * d_final * miu[j,i]
                d12 = d12 + delta_d
                i = i + iu
                ac = a_z
                a_z = a_z + azu
            else:
                delta_d = (a_y - ac) * d_final * miu[j,i]
                d12 = d12 + delta_d
                j = j + ju
                ac = a_y
                a_y = a_y + ayu
            if i >= nz or j >= ny or j < 0 or ac < 0:
                break

    # # 去掉 y=0 的第一行
    # data_pos = data[:, 1:, :]     # shape (nray, ny-1, nz)
    # data_zero = np.zeros((len(rays),ny-1,nz))
    # data_half = np.concatenate([data_zero, data], axis=1)
    # # data_pos = data
    # # 沿 y 维度翻转
    # data_mirror = np.flip(data_half, axis=1)  # 翻转 y 方向

    # # # 去掉 p=0 的第一行
    # data_mirror = data_mirror[1:, :, :]     # shape (nray-1, ny-1, nz)
    # data_mirror = np.flip(data_mirror, axis=0)  # 翻转 p 方向
    # # 拼接：先镜像部分(y<0)，再原部分(y>=0)
    # data_full = np.concatenate([data_mirror, data_half], axis=0)
    data_full = data
    # plt.imshow(data_full[4,:,:],cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    # ny = 2 * ny - 1 #odd
    # # ny = 2 * ny   #even
    # nray = 2 * nray - 1 #odd
    # nray = 2 * nray   #even
    p,m_y,m_z = np.nonzero(data_full)
    data = data_full[p,m_y,m_z]
    m = nz * m_y + m_z
    m_num = ny * nz
    p_num = nray
    # order = np.argsort(m)
    # p_idx = p[order]
    # data = data[order]
    # m_ptr = np.zeros(m_num+1,dtype=np.int32)
    # m_unique, counts = np.unique(m, return_counts=True)   
    # m_cnts = np.zeros(m_num); ptr = np.zeros(m_num+1); sum = 0
    # for k in range(m_unique.shape[0]):
    #     m_cnts[m_unique[k]] = counts[k]
    # for k in range(1, m_num+1):
    #     sum += m_cnts[k-1]
    #     m_ptr[k] = sum

    order = np.argsort(p)
    m_idx = m[order]
    data = data[order]
    p_ptr = np.zeros(p_num+1,dtype=np.int32)
    p_unique, counts = np.unique(p, return_counts=True)   
    p_cnts = np.zeros(p_num);sum = 0
    for k in range(p_unique.shape[0]):
        p_cnts[p_unique[k]] = counts[k]
    for k in range(1, p_num+1):
        sum += p_cnts[k-1]
        p_ptr[k] = sum

    vec = np.tile(ray_vec, (len(m_idx), 1))
    vec[:,[0,2]] = vec[:,[2,0]]

    return {
            "p_ptr": p_ptr,
            "m_idx": m_idx.astype(np.int32, copy=False),
            # "m_ptr": p_ptr,
            # "p_idx": m_idx.astype(np.int32, copy=False),
            "data": data.astype(np.float32, copy=False),
            "vec": vec.astype(np.float32, copy=False),
            "shape": (nray, m_num),
    }

    # return vector_full

def voxel_center_calulate(distance_of_source2object=40,object_size=[20,90],voxels_size=1):
    '''
    计算体素中心坐标(x,y,z)并返回体素中心矩阵,shape(ny,nz,3)
    distance_of_source2object: 源到物体前表面的距离
    object_size: 物体的实际尺寸(x=y,z) 单位mm
    voxels_size: 体素的大小 单位mm
    '''
    ny = math.floor(object_size[1] / voxels_size) + 1
    nz = math.floor(object_size[0] / voxels_size) + 1   
    voxel_center = np.zeros((ny,nz,3))

    for i in range(ny):
        for j in range(nz):
            voxel_center[i,j,:] = np.array([0,- 0.5 * object_size[1] + i * voxels_size,distance_of_source2object + j * voxels_size]) #x,y,z

    return voxel_center

def voxel_path_length_cal(grid_origin,grid_size,obj_size,ray_start,ray_end,miu = np.ones((40,40,40)),output_type="total"):
    '''
    计算体素路径长度
    input:
    grid_origin: 体素网格的原点坐标 [x,y,z]
    grid_size: 体素的大小 [x,y,z] 单位mm
    obj_size: 物体的大小 [x,y,z] 单位mm
    ray_start: 射线起始点 [x,y,z] 绝对坐标
    ray_end: 射线结束点 [x,y,z] 绝对坐标 start 和 end 一一对应,要求ray_end在体素网格外
    miu: 衰减系数矩阵[x,y,z] 单位 1/mm
    output_type: 输出类型 "single" 每条射线的路径长度和经过的体素坐标(调试用)  "total" 每条射线的总出射向量
    output:
        data: 每条射线的出射衰减

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

    grid_origin[[0,2]] = grid_origin[[2,0]]
    obj_size[[0,2]] = obj_size[[2,0]]
    grid_size[[0,2]] = grid_size[[2,0]]
    ray_start[:,[0,2]] = ray_start[:,[2,0]]
    ray_end[:,[0,2]] = ray_end[:,[2,0]]
    num = np.shape(ray_start)[0]
    # 用list动态存储
    vec_out_single = []
    vec_out_total = []
    nums, zs, ys, xs = [], [], [], []

    for num_i in range(num):
        print("processing ray:",num_i+1,"/",num,end="\r")
        soc = ray_start[num_i]

        nz = math.floor(obj_size[0] / grid_size[0])
        ny = math.floor(obj_size[1] / grid_size[1])
        nx = math.floor(obj_size[2] / grid_size[2])
        vec_intensity = 1
        vec = (ray_end[num_i] - ray_start[num_i])+[eps,eps,eps]
        d = np.linalg.norm(vec)
        vec = vec / d
        final_vec = soc + d * vec
        grid_z = grid_size[0]
        grid_y = grid_size[1]
        grid_x = grid_size[2]
        az = np.zeros(nz+1)
        ay = np.zeros(ny+1)
        ax = np.zeros(nx+1)
        if grid_origin[0]>0:
            for i in range(nz+1):
                az[i] = (grid_origin[0]+i*grid_z-soc[0]) / (final_vec[0] - soc[0] + eps)
        else:
            for i in range(nz+1):
                az[i] = (grid_origin[0]-i*grid_z-soc[0]) / (final_vec[0] - soc[0] + eps)
        for j in range(ny+1):
            ay[j] = (grid_origin[1]+j*grid_y-soc[1]) / (final_vec[1] - soc[1] + eps)
        for k in range(nx+1):
            ax[k] = (grid_origin[2]+k*grid_x-soc[2]) / (final_vec[2] - soc[2] + eps)

        az_min = min(az[0],az[nz])
        az_max = max(az[0],az[nz])
        ay_min = min(ay[0],ay[ny])
        ay_max = max(ay[0],ay[ny])
        ax_min = min(ax[0],ax[nx])
        ax_max = max(ax[0],ax[nx])

        a_min = max(az_min,ay_min,ax_min)
        a_max = min(az_max,ay_max,ax_max)
        # if output_type == "single":
        if soc[0]<final_vec[0]:
            if a_min == az_min:
                i_min = 1
            else:
                i_min = math.ceil((soc[0] + a_min * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            if a_max == az_max:
                i_max = nz
            else:
                i_max = math.floor((soc[0] + a_max * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_z)
            a_z = az[i_min]
            iu = 1
        else:
            if a_min == az_min:
                i_max = nz-1
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
                j_max = ny
            else:
                j_max = math.floor((soc[1] + a_max * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
            a_y = ay[j_min]
            ju = 1
        else:
            if a_min == ay_min:
                j_max = ny-1
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
                k_max = nx
            else:
                k_max = math.floor((soc[2] + a_max * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_x)
            a_x = ax[k_min]
            ku = 1
        else:
            if a_min == ax_min:
                k_max = nx-1
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
        azu = iu * grid_z / (final_vec[0]-soc[0]+eps)
        ayu = ju * grid_y / (final_vec[1]-soc[1]+eps)
        axu = ku * grid_x / (final_vec[2]-soc[2]+eps)
        ac = a_min

        i = first_i
        j = first_j
        k = first_k
        d12 = 0
        if ac >0:
            d12 = d * ac *  0.00001638 #air
        delta_d = 0 
        intensity = vec_intensity
        # for q in range(Np+1):
        data = np.zeros((nx,ny,nz))#调试可视化用
        while True:
            # i12 = intensity * np.exp(-miu[k,j,i] * d12)
            if a_z < a_y and a_z < a_x:
                i12 = intensity * np.exp(-d12)
                data[k,j,i] = i12
                nums.append(num_i)
                zs.append(i)
                ys.append(j)
                xs.append(k)
                vec_out_single.append(i12 * vec)
                if ac >0:
                    delta_d = (a_z - ac) * d * miu[k,j,i]
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                i = i + iu
                ac = a_z
                a_z = a_z + azu
            elif a_y < a_x and a_y < a_z:
                if ac >0:
                    delta_d = (a_y - ac) * d * miu[k,j,i]
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                j = j + ju
                ac = a_y
                a_y = a_y + ayu
            else:
                if ac >0:
                    delta_d = (a_x - ac) * d * miu[k,j,i]
                else:
                    delta_d = 0
                d12 = d12 + delta_d
                k = k + ku
                ac = a_x
                a_x = a_x + axu
            if i >= nz or i < 0 or j >= ny or j < 0 or k >= nx or k < 0:
                break
        
        # print("total voxels:",len(xs))
        i12 = intensity * np.exp(-d12)
        vec_out_total.append(i12 * vec)
        # if output_type == "total":
        #     intensity = 1
        #     d12 = d * a_max * 0.001
        #     i12 = intensity * np.exp(-miu[k,j,i] * d12)
        #     vec_out_total.append(i12 * vec)
        # data_show = np.sum(data, axis=2)
        # plt.imshow(data_show, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
    if output_type == "single":
        vec_out = np.array(vec_out_single).astype(np.float32, copy=False)
        vec_out[[0,2]] = vec_out[[2,0]]
        return {
                "num": np.array(nums).astype(np.int32, copy=False),
                "z": np.array(zs).astype(np.int32, copy=False),
                "x": np.array(xs).astype(np.int32, copy=False),
                "y": np.array(ys).astype(np.int32, copy=False),
                "vec": vec_out
        }
    if output_type == "total":
        vec_out = np.array(vec_out_total).astype(np.float32, copy=False)
        vec_out[[0,2]] = vec_out[[2,0]]
        data = np.linalg.norm(vec_out,axis=1,keepdims=False)
        data = np.array(data, dtype=np.float32)
        return data

def main():
    distance_of_source2object=5
    object_size=np.array([40,40,40])
    ny=40
    nz=40
    ray_angle=90*np.pi/180
    ray_step=15*np.pi/180
    voxels_size=np.array([1,1,1])
    attenuation1 = 0.068 * np.ones((ny,nz))
    attenuation2 = 0.034 * np.ones((ny,ny,nz))
    grid_origin = np.array([-0.5 * object_size[0], -0.5 * object_size[1], distance_of_source2object])
    grid_size = np.array([1, 1, 1])
    ray_start = np.array([[0,0,0],[0,0,0],[0,0,25],[5,5,25]])
    ray_end = np.array([[0,0,60],[-5,5,60],[5,-5,60],[-5,-5,0]])
    result = incident_vector_calulate(distance_of_source2object,object_size[1:],ny,nz,ray_angle,ray_step,voxels_size,attenuation1)
    result2 = voxel_path_length_cal(grid_origin,grid_size,object_size,ray_start,ray_end,attenuation2,output_type="total")
    print("done")

if __name__ == "__main__":
    main()