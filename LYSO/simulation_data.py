import os
import numpy as np
import matplotlib.pyplot as plt
import math
import IntersectionLength as IL
from generate_detectors_yaml import generate_yaml
eps = 1e-6

def ScatterEnergy(Energy,costheta):
    '''Calculate scattered photon energy after Compton scattering.
    Energy : Incident photon energy in keV
    costheta : Cosine of scattering angle'''
    E = np.array(Energy)/(1+np.array(Energy)*(1-np.array(costheta))/511)
    return E

def Klein_Nishina(Energy,costheta):
    '''Calculate differential cross section using Klein-Nishina formula.
    Energy : Incident photon energy in keV
    costheta : Cosine of scattering angle'''
    E = ScatterEnergy(Energy,costheta)
    r0 = 2.818e-13 # cm
    d_sigma = (E/Energy)**2*(E/Energy + Energy/E + (1-costheta**2))#*0.5*r0**2
    return d_sigma

def CosTheta_cal(voxel_pos, detector_pos, ray_incident):
    """
    calculate scatter cos(theta)
    voxel_pos: (N_voxel, 3)
    detector_pos: (N_det, 3)
    ray_incident: (1, 3) 
    return:
    Costheta: (N_voxel * N_det,)
    """
    N_voxel = voxel_pos.shape[0]
    N_det = detector_pos.shape[0]

    voxel_pos_expand = np.tile(voxel_pos, (N_det, 1))
    detector_pos_expand = np.repeat(detector_pos, N_voxel, axis=0)
    
    ray_scatter = detector_pos_expand - voxel_pos_expand
    norm = np.linalg.norm(ray_scatter, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    ray_scatter_norm = ray_scatter / norm
    
    ray_incident = np.repeat(ray_incident[None,:], N_voxel * N_det, axis=0)

    Costheta = np.sum(ray_incident * ray_scatter_norm, axis=1)
    
    return Costheta

def Omega_cal(start,end_center,end_size,mode="approx"):
    '''Calculate solid angle of a detector pixel from a grid point.
    start: coordinate of the grid point
    end_center: coordinate of the center of the detector pixel
    end_size: size of the detector pixel
    '''
    start_num = start.shape[0]
    end_num = end_center.shape[0]
    start = np.tile(start, (end_num, 1))
    end_center = np.repeat(end_center, start_num, axis=0)
    if mode == "approx":
        r = np.linalg.norm(end_center - start,axis=1)
        A = end_size**2
        Omega = A/r**2
    elif mode == "exact":
        # 叉积求立体角：
        p1 = end_center + np.array([-end_size/2, -end_size/2]) - start
        p2 = end_center + np.array([ end_size/2, -end_size/2]) - start
        p3 = end_center + np.array([ end_size/2,  end_size/2]) - start
        p4 = end_center + np.array([-end_size/2,  end_size/2]) - start
        n1 = np.cross(p1,p2)
        n2 = np.cross(p2,p3)
        n3 = np.cross(p3,p4)
        n4 = np.cross(p4,p1)
        Omega = np.arctan2(np.linalg.norm(n1), np.dot(n1, n2)) + np.arctan2(np.linalg.norm(n3), np.dot(n3, n4)) + np.arctan2(np.linalg.norm(n2), np.dot(n2, n3)) + np.arctan2(np.linalg.norm(n4), np.dot(n4, n1))
    return Omega

def incident_decay(voxel_size,nz,attenuation):
    '''Calculate incident photon decay along z direction in the grid.
    voxel_size: size of the voxel [y,z]
    nz: number of grid points in z direction
    attenuation: attenuation coefficient in mm^-1'''
    dz = voxel_size[1] / nz
    decay = np.zeros((nz))
    for iz in range(nz):
        path_length = (iz + 0.5) * dz
        decay[iz] = np.exp(-attenuation * path_length)
    return decay

def scatter_decay(grid_origin,grid_size,obj_size,ray_start,ray_end,miu = np.ones((40,40,40)),output_type="total"):
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

    # grid_origin[[0,2]] = grid_origin[[2,0]]
    # obj_size[[0,2]] = obj_size[[2,0]]
    # grid_size[[0,2]] = grid_size[[2,0]]
    # ray_start[:,[0,2]] = ray_start[:,[2,0]]
    # ray_end[:,[0,2]] = ray_end[:,[2,0]]
    num = np.shape(ray_start)[0]
    # 用list动态存储
    vec_out_single = []
    vec_out_total = []
    nums, zs, ys, xs = [], [], [], []

    for num_i in range(num):
        print("processing ray:",num_i+1,"/",num,end="\r")
        soc = ray_start[num_i]

        nx = math.floor(obj_size[0] / grid_size[0])
        ny = math.floor(obj_size[1] / grid_size[1])
        nz = math.floor(obj_size[2] / grid_size[2])
        vec_intensity = 1
        vec = (ray_end[num_i] - ray_start[num_i])+[eps,eps,eps]
        d = np.linalg.norm(vec)
        vec = vec / d
        final_vec = soc + d * vec
        grid_x = grid_size[0]
        grid_y = grid_size[1]
        grid_z = grid_size[2]
        az = np.zeros(nz+1)
        ay = np.zeros(ny+1)
        ax = np.zeros(nx+1)
        if grid_origin[2]>0:
            for i in range(nz+1):
                az[i] = (grid_origin[2]+i*grid_z-soc[2]) / (final_vec[2] - soc[2] + eps)
        else:
            for i in range(nz+1):
                az[i] = (grid_origin[2]-i*grid_z-soc[2]) / (final_vec[2] - soc[2] + eps)
        for j in range(ny+1):
            ay[j] = (grid_origin[1]+j*grid_y-soc[1]) / (final_vec[1] - soc[1] + eps)
        for k in range(nx+1):
            ax[k] = (grid_origin[0]+k*grid_x-soc[0]) / (final_vec[0] - soc[0] + eps)

        az_min = min(az[0],az[nz])
        az_max = max(az[0],az[nz])
        ay_min = min(ay[0],ay[ny])
        ay_max = max(ay[0],ay[ny])
        ax_min = min(ax[0],ax[nx])
        ax_max = max(ax[0],ax[nx])

        a_min = max(az_min,ay_min,ax_min)
        a_max = min(az_max,ay_max,ax_max)
        # if output_type == "single":
        if soc[2]<final_vec[2]:
            if a_min == az_min:
                i_min = 1
            else:
                i_min = math.ceil((soc[2] + a_min * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_z)
            if a_max == az_max:
                i_max = nz
            else:
                i_max = math.floor((soc[2] + a_max * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_z)
            a_z = az[i_min]
            iu = 1
        else:
            if a_min == az_min:
                i_max = nz-1
            else:
                i_max = math.floor((soc[2] + a_min * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_z)
            if a_max == az_max:
                i_min = 0
            else:
                i_min = math.ceil((soc[2] + a_max * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_z)
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

        if soc[0]<final_vec[0]:
            if a_min == ax_min:
                k_min = 1
            else:
                k_min = math.ceil((soc[0] + a_min * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_x)
            if a_max == ax_max:
                k_max = nx
            else:
                k_max = math.floor((soc[0] + a_max * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_x)
            a_x = ax[k_min]
            ku = 1
        else:
            if a_min == ax_min:
                k_max = nx-1
            else:
                k_max = math.floor((soc[0] + a_min * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_x)
            if a_max == ax_max:
                k_min = 0
            else:
                k_min = math.ceil((soc[0] + a_max * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_x)
            a_x = ax[k_max]
            ku = -1

        Np = i_max - i_min + j_max - j_min + k_max - k_min + 3

        first_i = math.floor((soc[2]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[2]-soc[2]) - grid_origin[2]) / grid_z)
        first_j = math.floor((soc[1]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[1]-soc[1]) - grid_origin[1]) / grid_y)
        first_k = math.floor((soc[0]+0.5 * (min(a_z,a_y,a_x) + a_min) * (final_vec[0]-soc[0]) - grid_origin[0]) / grid_x)
        azu = iu * grid_z / (final_vec[2]-soc[2]+eps)
        ayu = ju * grid_y / (final_vec[1]-soc[1]+eps)
        axu = ku * grid_x / (final_vec[0]-soc[0]+eps)
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
        # vec_out[[0,2]] = vec_out[[2,0]]
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

def mlem_reconstruction(sys_mat, detector_data, n_iter=50):
    # 初始化重建结果为全1
    n_voxel = sys_mat.shape[1]
    rec = np.ones(n_voxel, dtype=np.float32)

    # 计算系统矩阵的列和（防止除0）
    sys_sum = np.sum(sys_mat, axis=0)
    sys_sum[sys_sum == 0] = 1e-8

    for k in range(n_iter):
        # 前向投影
        forward = sys_mat @ rec
        forward[forward == 0] = 1e-8  # 避免除零

        # 比值项（测量/预测）
        ratio = detector_data / forward

        # 反投影更新项
        backproj = sys_mat.T @ ratio

        # 更新重建结果
        rec *= backproj / sys_sum

        # 可选：打印每次迭代的进度
        if (k + 1) % 10 == 0:
            print(f"Iteration {k+1}/{n_iter} completed")

    return rec

def main():
    # system parameters
    Energy = 160  # keV
    scatter_angle = 90  # degrees
    theta_0 = (180 - scatter_angle) * math.pi / 180  
    attenu_obj = 0.001  # mm^-1
    attenu_det = 7.1 * 0.97 * 1e-3  # mm^-1
    ray_incident = np.array([0, 0, 1])  # Incident ray along z-axis
    source_pos = np.array([0, 0, 0])
    obj_size = np.array([4, 4, 20])  # mm
    source2object = 10  # mm
    source2point = 25  # mm
    point_pos = source_pos + np.array([0, 0, source2point])
    grid_start = np.array([-obj_size[0]/2, -obj_size[1]/2, source2object])
    grid_end = np.array([ obj_size[0]/2, obj_size[1]/2, source2object + obj_size[2]])
    nx,ny,nz = 4,4,100
    dx = (grid_end[0]-grid_start[0])/nx
    dy = (grid_end[1]-grid_start[1])/ny
    dz = (grid_end[2]-grid_start[2])/nz
    grid_size = np.array([dx,dy,dz])
    voxel_pos = np.zeros((nz, 3))
    for iz in range(nz):
        idx = iz
        y_pos = 0
        z_pos = grid_start[2] + (iz + 0.5) * dz
        voxel_pos[idx,:] = np.array([0, y_pos, z_pos])

    # voxel_pos = np.reshape(voxel_pos, (nz, 3))

    # incident_data = incident_decay([dy,dz],nz,attenu_obj)

    point2det = 30  # mm
    detector_size_x = 48  # mm
    n_det = 48
    cell_size = detector_size_x / n_det
    det0_size = np.array([detector_size_x, cell_size, 10])  # mm long detector
    det1_size = np.array([detector_size_x, cell_size, 5])  # mm short detector
    det_sizes = np.zeros((n_det,3))
    for i in range(n_det):
        if i % 2 == 0:
            det_sizes[i,:] = det0_size
        else:
            det_sizes[i,:] = det1_size

    det_zvec = np.array([0, math.sin(theta_0), - math.cos(theta_0)]) # detector normal vector
    det_yvec = np.array([0, math.cos(theta_0), math.sin(theta_0)])
    det_xvec = np.cross(det_zvec, det_yvec) # [1,0,0]
    detector_center = point_pos + det_zvec * point2det

    detector_type = np.linspace(1 ,n_det, n_det)
    detector_type[detector_type % 2 == 0]  = 0  # long
    detector_type[detector_type % 2 == 1]  = 1  # short
    detector_pos = np.zeros((n_det, 3))
    for i in range(n_det):
        offset = (i - (n_det - 1) / 2) * cell_size
        detector_pos[i,:] = detector_center + det_yvec * offset + det_zvec * ((det0_size[2]/2) - detector_type[i] * (det0_size[2]-det1_size[2])/2)
    
    cell_x = np.linspace(-detector_size_x/2, detector_size_x/2, n_det)
    det_cell_pos = np.zeros((n_det, n_det, 3))
    for i in range(n_det):
        offset = cell_x[i]
        det_cell_pos[i,:,:] =  det_xvec * offset + detector_pos
    det_cell_pos = np.reshape(det_cell_pos, (n_det * n_det, 3))

    # ray_start = np.tile(voxel_pos,(det_cell_pos.shape[0],1))
    # ray_end = np.repeat(det_cell_pos,voxel_pos.shape[0],axis=0)
    # scatter_data = scatter_decay(grid_start,grid_size,obj_size,ray_start,ray_end,miu = attenu_obj * np.ones((nx,ny,nz)),output_type="total")

    # Costheta = CosTheta_cal(voxel_pos, det_cell_pos, ray_incident)
    # kn_data = Klein_Nishina(Energy, Costheta)
    # Omega = Omega_cal(voxel_pos, det_cell_pos, cell_size, mode="approx")

    # generate_yaml(detector_pos, det_xvec, det_yvec, det_zvec, det_sizes, mu=0.97, rho=7.1, filename="detectors.yaml")
    
    # Cosalpha = CosTheta_cal(voxel_pos, detector_pos, det_zvec)
    # mask = np.zeros(Cosalpha.shape)
    # mask(Cosalpha > 1-eps) = 1
    tool = IL.CalIntersectionLength("./LYSO/detectors.yaml")
    lengths_nk = np.zeros((n_det, det_cell_pos.shape[0]), dtype=np.float32)

    for i in range(voxel_pos.shape[0]):
        start = voxel_pos[i, :]
        lengths_nk = tool.calLength(start, det_cell_pos, normals=tool.ns, centers=tool.cs)  # [k = n_det, n_ray = n_det*n_det]
        print(i)
    #     deposit_nk = 1 - np.exp(-1 * attenu_det * lengths_nk)  # (n_det, n_ray)
    #     # deposit_k[:, i] = np.sum(deposit_nk, axis=1)  # (n_det, nz)
    #     Costheta = CosTheta_cal(start[None, :], det_cell_pos, ray_incident)
    #     kn_data = Klein_Nishina(Energy, Costheta)
    #     Omega = Omega_cal(start[None, :], det_cell_pos, cell_size, mode="approx")
    #     # scatter_data = scatter_decay(grid_start,grid_size,obj_size,np.tile(start, (det_cell_pos.shape[0], 1)),det_cell_pos,miu = attenu_obj * np.ones((nx,ny,nz)),output_type="total")
    #     sys_nk = deposit_nk * kn_data * Omega #* incident_data[i] #* scatter_data
    #     sys_mat[:,i] = np.sum(sys_nk, axis=1)  # (n_det, nz)
    #     print(f"point2det = {point2det}, mean(lengths_nk) = {np.mean(lengths_nk):.4e}, nonzero = {np.count_nonzero(lengths_nk)}")
    
    # sys_mat = np.tile(incident_data,n_det * n_det) * scatter_data * kn_data * Omega # (n_det * n_det , nz)
    # sys_mat = np.sum(sys_mat.reshape(n_det, n_det, nz), axis=0) * deposit_k # (n_det, nz)
    # sys_mat = np.reshape(sys_mat, (n_det,nz))
    # image_data = np.ones((nz,), dtype=np.float32)
    # image_data[5:13] = 0
    # detector_data = sys_mat @ image_data #* detector_type

    # rec = mlem_reconstruction(sys_mat, detector_data, n_iter=50)

    # # rec = sys_mat.T @ detector_data
    # plt.figure(figsize=(8, 6))
    # # plt.spy(sys_mat, markersize=1)
    # plt.imshow(sys_mat, cmap='gray', aspect='auto')
    # plt.title("Sparsity Pattern of System Matrix")
    # plt.xlabel("Voxel index (n)")
    # plt.ylabel("Ray index (m)")
    # plt.show()

    # plt.plot(detector_data)
    # plt.xlabel("Detector Pixel")
    # plt.ylabel("Intensity")
    # plt.title("Detector Response")
    # plt.show()

    # plt.plot(rec)
    # plt.xlabel("Voxel Index")
    # plt.ylabel("Intensity")
    # plt.title("Voxel value")
    # plt.show()

if __name__ == "__main__":
    main()