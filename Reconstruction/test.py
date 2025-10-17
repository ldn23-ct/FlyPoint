import numpy as np
from scipy.ndimage import zoom
import incident as Inc
import matplotlib.pyplot as plt
from tqdm import tqdm
import IntersectionLength as InL

def Mapping(pos: np.ndarray, objsize: np.ndarray, voxelsize: np.ndarray, kernelsize: np.ndarray):
    '''
    input
        pos--模体前表面左上角点坐标 初始时刻对准零点 ndarray of shape (2,) dtype=float
        objsize--物体尺寸 ndarray of shape (3,)
        voxelsize--空间体素大小 根据空间坐标计算体素编号
        kernel_y/kernel_z--二维卷积核尺寸 大小与体素大小相同 计算会极大简化
    output
        n_grid--二维卷积核每个像素与三维空间体素的对应关系 shape (ky, kz)
    idea
        (1) 空间体素划分编号 n = iz + iy * nz + ix * (ny * nz)
        (2) 零点按模体中心来计算
        (3) 假设零点是对准的 kernel相当于三维矩阵的一个切片 只需要根据 pos_y 就可以求出 iy0
            并顺便取出每一行对应的 iy 
        (4) 当 pos_x 位于交界线上 归 x+ 像素
        (5) 越界 n_grid赋值-1
    '''
    x, y = pos[0], pos[1]
    Lx, Ly, Lz = objsize[0], objsize[1], objsize[2]
    dx, dy, dz = voxelsize[0], voxelsize[1], voxelsize[2]
    ky, kz = kernelsize[0], kernelsize[1]
    nx, ny, nz = Lx/dx, Ly/dy, Lz/dz

    idx = int(np.floor((x + Lx/2) / dx))
    if idx < 0 or idx > nx:
        return -np.ones((ky*kz,), dtype=np.int64)
    if idx == nx: idx = idx - 1

    idy_center = int(np.floor((Ly/2 - y) / dy))
    idy_grid = idy_center + (np.arange(ny) - ky // 2)[:, None]
    idz_grid = np.arange(nz)[None, :]
    in_y = (idy_grid >= 0) & (idy_grid < ny)
    in_z = (idz_grid >= 0) & (idz_grid < nz)
    mask = in_y & in_z

    n_grid = idz_grid + idy_grid * nz + idx * (ny * nz)
    n_grid = np.where(mask, n_grid, -1)

    return n_grid

def DetArray(corners: list, pixelsize: list, detsize: list):
    '''
    input
        corner_pos--4 corners position of the detector  左上角起点，逆时针排序
        dx/dy--length of the detector pixel
        x/y--length of the detector
    output
        detarray--pos of the pixels' center shape:[ny, nx]
    idea
        (1) 计算像素个数
        (2) 计算每个像素中心点
    '''
    print("enter detarray")
    nx, ny = int(detsize[0] / pixelsize[0]), int(detsize[1] / pixelsize[1])
    P0, P1, P2, P3 = map(np.array, corners)
    centers = []

    n = np.cross(P1 - P0, P2 - P0)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise ValueError("collinear vertices")

    for i in tqdm(range(ny)):
        v_top_start = P0 + (P3 - P0) * i / ny
        v_top_end   = P1 + (P2 - P1) * i / ny
        v_bot_start = P0 + (P3 - P0) * (i + 1) / ny
        v_bot_end   = P1 + (P2 - P1) * (i + 1) / ny
        for j in range(nx):
            # Compute 4 corners of the sub-rectangle
            tl = v_top_start + (v_top_end - v_top_start) * j / nx        # top-left
            br = v_bot_start + (v_bot_end - v_bot_start) * (j + 1) / nx  # bottom-right
            center = (tl + br) / 2  # diagonal midpoint
            centers.append(center)
    centers = np.array(centers).reshape((nx, ny, 3))
    centers = (centers.transpose(1, 0, 2)).reshape((-1, 3))
    return n/nn, centers

class ReConstruction:
    '''
    '''
    def __init__(self,
                 src_pos: np.ndarray,  #shape: [3,]
                 grid_origin: np.ndarray,  #shape: [3,]
                 obj_size: np.ndarray,  #shape: [3,]
                 voxelsize: np.ndarray,  #shape: [3,]
                 det_corners: np.ndarray,  #shape: [4, 3]  points in order
                 det_size: np.ndarray,  #shape: [2,]
                 pixel_sizeL: np.ndarray,  #shape: [2,]
                 pixel_sizeS: np.ndarray,  #shape:[2,]
                 fan_angle: np.float32,
                 E: np.ndarray,  # shape: [e,]
                 prob: np.ndarray,  # shape: [e,]
                 rho: np.ndarray  # shape: ?
                 ):
        self.src = src_pos
        self.obj_origin = grid_origin
        self.objsize = obj_size
        self.voxelsize = voxelsize
        self.detsize = det_size
        self.pixelsizeL = pixel_sizeL
        self.pixelsizeS = pixel_sizeS
        self.detcorners = det_corners
        # self.detnormal, self.det = DetArray(det_corners, self.pixelsize, self.detsize)  # 按行排序，右下角为起点
        self.fan = np.deg2rad(fan_angle)
        self.E = E
        self.prob = prob / np.sum(prob)
        self.rho = rho
    
    def Emit(self):
        SOD = np.abs(self.obj_origin[2] - self.src[2])
        slice_halfy = (SOD + self.objsize[2]) * np.tan(self.fan / 2)
        ny = int(2 * np.ceil((slice_halfy - self.voxelsize[1]/2) / self.voxelsize[1])) + 1
        nz = int(np.ceil(self.objsize[2] / self.voxelsize[2]))
        obj_slice_size = [ny * self.voxelsize[1], self.objsize[2]]
        self.ny, self.nz = ny, nz
        #------------------ slice sample ------------------#
        vec = -1  #  物体从负方向开始
        y_start = (obj_slice_size[0] - self.voxelsize[1]) / 2
        z_start = self.obj_origin[2] + vec * self.voxelsize[2] / 2
        y_centers = [(y_start - i * self.voxelsize[1]) for i in range(ny)]
        z_centers = [(z_start + i * vec * self.voxelsize[2]) for i in range(nz)]
        Y, Z = np.meshgrid(y_centers, z_centers, indexing='ij')
        X = np.zeros_like(Y)
        self.obj_slice = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # 按行排序, 左上角为起点
        #------------------ slice sample ------------------#
        # self.emit_data = Inc.incident_vector_calulate(SOD,
        #                                         obj_slice_size,
        #                                         ny,
        #                                         nz,
        #                                         self.fan, 
        #                                         ray_step=np.deg2rad(0.1),
        #                                         voxels_size=self.voxelsize,
        #                                         miu=self.rho[0])
        # np.save("./data/emit_pptr.npy", self.emit_data["p_ptr"])
        # np.save("./data/emit_midx.npy", self.emit_data["m_idx"])
        # np.save("./data/emit_data.npy", self.emit_data["data"])
        # np.save("./data/emit_vec.npy", self.emit_data["vec"])

        #------------------ test-- 1 ray ------------------#
        # ny = 1
        # nz = int(np.ceil(self.objsize[2] / self.voxelsize[2]))
        # self.ny, self.nz = ny, nz   
        # SOD = np.abs(self.obj_origin[2] - self.src[2])
        # vec = -1
        # z_start = (SOD + self.voxelsize[2] / 2) * vec
        # y_centers = [0]*ny
        # z_centers = [(z_start - i * self.voxelsize[2]) for i in range(nz)]    # 物体在z负方向
        # Y, Z = np.meshgrid(y_centers, z_centers, indexing='ij')
        # X = np.zeros_like(Y)             
        # self.obj_slice = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # 按行排序, 左上角为起点
        # p_ptr = np.array([0, nz])
        # m_idx = np.arange(nz)
        # vec = np.zeros((nz, 3))
        # vec[:, 2] = 1
        # data = np.ones((nz,))
        # self.emit_data = {
        #     "p_ptr": p_ptr,
        #     "m_idx": m_idx.astype(np.int32),
        #     "data": data.astype(np.float64),
        #     "vec": vec.astype(np.float64),
        #     "shape": (1, nz)
        # }
        #------------------ test-- 1 ray ------------------#

    def Scatter(self):
        hc = HalfSpaceCutting()
        ns_slit, rs_slit = hc.build_six_faces(self.slit[0:4], self.slit[4:])
        p, q = hc.loda_point(self.obj_slice, self.det, rs_slit)
        self.scatter_data = hc.through_slit(p, q, ns_slit)
        
        #------------------ calculate solid angle and decay ------------------#
        vec = self.scatter_data["vec"]
        start = self.obj_slice[self.scatter_data["m_idx"]]
        end = self.det[self.scatter_data["n_idx"]]
        r = np.linalg.norm(start - end, axis=1)
        
        cos_phi = np.linalg.norm(vec - np.dot(vec, ns_slit[0])[:, None] * ns_slit[0])
        solid_angle = self.pixelsize[0]*self.pixelsize[1] * cos_phi / r
        
        # 这里需要计算模体的一个角点空间坐标传入函数
        objcorner = -1 * self.objsize / 2
        objcorner[2] = self.obj_origin[2] - self.src[2]
        start_point = self.obj_slice[self.scatter_data["m_idx"]]
        end_point = self.det[self.scatter_data["n_idx"]]
        # decay = np.load("./decay.npy")
        # decay = Inc.voxel_path_length_cal(objcorner, self.voxelsize, self.objsize, start_point, end_point, attenuation=0)
        # np.save("decay.npy", decay)
        # print("decaydata has been saved")
        decay = np.ones((solid_angle.shape[0]))
        assert solid_angle.shape[0] == decay.shape[0]
        # decay = 1
        self.scatter_data["data"] = solid_angle * decay
        #------------------ calculate solid angle and decay ------------------#
 
    def ScatterM(self):
        # inl = InL.CalIntersectionLength(path="./Reconstruction/Vertex.yaml")
        # _, detS = DetArray(self.detcorners, self.pixelsizeS, self.detsize)
        # self.scatter_data = inl.through_slit(obj_array=self.obj_slice,
        #                                      det_array=detS,
        #                                      scale = int(self.pixelsizeL[0] / self.pixelsizeS[0]),
        #                                      threshold=1000,
        #                                      save=True)  # [m, n] [7e3, 2.5e3]
        self.scatter_data = np.load("./data/scatter_data.npy")
        # print(np.min(tool.scatter_data), np.max(tool.scatter_data))
        #------------------ calculate solid angle and decay ------------------#
        det_normal, self.detL = DetArray(self.detcorners, self.pixelsizeL, self.detsize)
        x, y = np.arange(self.obj_slice.shape[0]), np.arange(self.detL.shape[0])
        m_idx, n_idx = np.meshgrid(x, y, indexing='ij')
        start = self.obj_slice[m_idx]
        end = self.detL[n_idx]
        #------------------ calculate one decay ------------------#
        # emit_pptr = self.emit_data["p_ptr"]
        # emit_midx = self.emit_data["m_idx"]
        # l, r = emit_pptr[0], emit_pptr[1]
        # x, y = emit_midx[l:r], np.arange(self.detL.shape[0])
        # m_idx, n_idx = np.meshgrid(x, y, indexing='ij')
        # start_point = self.obj_slice[m_idx].reshape(-1, 3)
        # end_point = self.detL[n_idx].reshape(-1, 3)
        #------------------ calculate one decay ------------------#
        vec = (end - start).reshape(-1, 3)
        r = np.linalg.norm(vec, axis=1)
        vec = vec / r[:, None]
        self.scatter_vec = vec.reshape((self.obj_slice.shape[0], self.detL.shape[0], 3))
        
        cos_phi = np.sqrt(1 - (np.clip(vec@det_normal, -1.0, 1.0))**2)
        solid_angle = self.pixelsizeL[0]*self.pixelsizeL[1] * cos_phi / np.square(r)
        # 立体角很小，乘以系数因子，整体进行缩放
        scale = 1 / np.max(solid_angle)
        solid_angle = scale * solid_angle
        
        # objcorner = -1 * self.objsize / 2
        # objcorner[2] = self.obj_origin[2]
        # decay = Inc.voxel_path_length_cal(objcorner, self.voxelsize, self.objsize, start_point, end_point, self.rho)
        # np.save("./data/scatter_decay_angle0.npy", decay)
        
        self.scatter_data = self.scatter_data.reshape((self.obj_slice.shape[0], self.detL.shape[0]))
        # self.scatter_data = (solid_angle * self.scatter_data).reshape((self.obj_slice.shape[0], self.detL.shape[0]))
        #------------------ calculate solid angle and decay ------------------#
 
    def klein_nishina(self, mu):
        """
        Klein-Nishina 微分散射截面
        输入:
            E   : 入射光子能量 keV  shape: [e,]
            prob: 不同能量光子的发射概率, 已归一化  shape: [e,]
            mu  : 散射角余弦 cos(theta)  shape: [n,]
            unit: "keV" 或 "MeV"(默认 "keV")
        输出:
            dsdo: 微分截面 dσ/dΩ,单位 mm^2/sr   shape: [n,]
        公式:
            E'/E = 1 / ( 1 + (E/mec2)*(1 - mu) )
            dσ/dΩ = (re^2/2) * (E'/E)^2 * ( E'/E + E/E' - sin^2θ )
                = (re^2/2) * (κ^2) * ( κ + 1/κ - (1 - mu^2) ), 其中 κ = E'/E
        """
        E = np.asarray(self.E, dtype=np.float64)
        prob = np.asarray(self.prob, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        E_grid  = np.atleast_1d(E)[:,  None]
        mu_grid = np.atleast_1d(mu)[None, :]
        # 常数（以 cm、keV 为基准）
        re_cm = 2.8179403262e-13        # 经典电子半径 [cm]
        mec2_keV = 511.0                # 电子静能 [keV]

        # 计算 E'/E
        k = 1.0 / (1.0 + (E_grid / mec2_keV) * (1.0 - mu_grid))   # κ = E'/E
        invk = 1.0 / k
        # sin^2 θ = 1 - mu^2
        sin2 = 1.0 - mu_grid**2
        # Klein–Nishina 微分截面 [cm^2/sr]
        # dsdo = 0.5 * (re_cm**2) * (k**2) * (k + invk - sin2) * 100  # shape: [e, n]
        dsdo = 0.5 * (k**2) * (k + invk - sin2)  # re_cm是常数，为了保证数值精度，整体进行缩放
        dsdo = np.sum(dsdo * prob[:, None], axis=0)
        return dsdo
        
    def Cal_SysMatrix(self, save=True):
        A_ptr = self.emit_data["p_ptr"]; Am_idx = self.emit_data["m_idx"]
        A_data = self.emit_data["data"]; A_vec = self.emit_data["vec"]
        P = A_ptr.shape[0] - 1
        M = self.scatter_data.shape[0]
        N = self.detL.shape[0]
        
        for p in tqdm(range(P)):
            sys_matrix_p = np.zeros((M, N))
            per_matrix_p = np.zeros((M, N))
            al, ar = A_ptr[p], A_ptr[p+1]
            m_idx = Am_idx[al:ar]
            emit_vec, emit_data = A_vec[al:ar], A_data[al:ar]  # [m_p, 3]  [m_p,]
            # scatter_vec, scatter_data
            scatter_vec = self.scatter_vec[m_idx, :, :]  # [m_p, 2.5e3, 3]
            costheta = np.einsum('mc, mnc->mn', emit_vec, scatter_vec)  # [m_p, 2.5e3]
            np.clip(costheta, -1.0, 1.0, out=costheta)
            
            scatter_data = self.scatter_data[m_idx, :]  # [m_p, 2.5e3]
            coffei = emit_data[:, None] * scatter_data

            kn = self.klein_nishina(costheta.ravel()).reshape((m_idx.shape[0], N))
            # print(np.min(emit_data), np.max(emit_data))
            # print(np.min(scatter_data), np.max(scatter_data))
            print(np.min(coffei[350:360]), np.max(coffei[350:360, :]))
            print(np.min(kn[350:360, :]), np.max(kn[350:360, :]))
            coffei = coffei * kn  # [m_p, 2.5e3]
            print(np.min(coffei[350:360, :]), np.max(coffei[350:360, :]))
            per = coffei / np.sum(coffei, axis=0)[None, :]
            value = per / coffei
            for i in range(coffei.shape[0]):
                sys_matrix_p[m_idx[i], :] = value[i, :]
                per_matrix_p[m_idx[i], :] = per[i, :]
            if save:
                np.save(f"./sys_matrix/per{p:03d}.npy", per_matrix_p)
                np.save(f"./sys_matrix/decay{p:03d}.npy", sys_matrix_p)
            
    def BackProjection(self, per: np.ndarray, sys_matrix: np.ndarray, det_response: np.ndarray):
        '''
        per: shape [m, n]
        sys_matrix: shape [m, n]
        det_response: shape [n,]
        '''
        # 只取贡献值在最大贡献值10%以上的体素计算
        per_max = np.max(per, axis=0)
        mask = (per / per_max[None, :]) > 0.1
        sum_num = np.sum(mask, axis=1)
        sum_num[np.isclose(sum_num, 0)] = 1
        result = sys_matrix * det_response[None, :]
        vox = np.sum(result * mask, axis=1) / sum_num
        
        # result = np.sum(sys_matrix * det_response[None, :], axis=1)
        proj = vox.reshape((int(vox.shape[0]/700), 700))
        # plt.imshow(proj, cmap="gray", vmin=np.min(result), vmax=np.max(result), aspect='auto')
        
        z = np.sum(proj, axis=0)
        x =  np.arange(z.shape[0])
        plt.plot(x, z)
        plt.show()
        return z
            

if __name__ == "__main__":
    src = np.array([0, 0, 158])
    obj_origin = np.array([0, 0, -50])
    objsize = np.array([200, 200, 70])
    fan = 15
    voxelsize = np.array([5, 5, 0.1])
    det_size = np.array([50, 50])
    pixelsizeS = np.array([0.1, 0.1])
    pixelsizeL = np.array([1, 1])
    det_corners = [[-58.19, 25, 52.51], [-58.19, -25, 52.51], [-90.33, -25, 14.2], [-90.33, 25, 14.2]]
    slitcorners = np.array([[-27.65, 25, 194.59], [-27.65, -25, 194.59], [-28.3, -25, 195.35], [-28.3, 25, 195.35],
                   [-27.95, 25, 193.7], [-27.95, -25, 193.7], [-28.91, -25, 194.85], [-28.91, 25, 194.85]])
    E = np.array([160])
    prob = np.array([1])
    # rho = 1
    voxelshape = (objsize / voxelsize).astype(np.int32)
    rho = np.zeros(voxelshape)
    rho.fill(0.134*2.7*1e-3)
    
    detResponses = np.load("./data/scan_results_ray_centric.npy")
    # detResponses = np.load("./data/results.npy")
    detResponses_nodefect = np.load("./data/no_defect.npy")
    detResponse = detResponses[75]
    detResponse_nodefect = detResponses_nodefect[75]
    # detResponse = np.load("./simulated_detector_image.npy")
    # detResponse = zoom(detResponse, zoom=10, order=3)
    # detResponse = np.ones((500, 500))
    # print(detResponse.shape)
    # plt.imshow(detResponse, cmap="gray", vmin=np.min(detResponse), vmax=np.max(detResponse), aspect='auto')
    # plt.show()
    detResponse = detResponse.ravel()
    detResponse_nodefect = detResponse_nodefect.ravel()
    
    
    
    tool = ReConstruction(src_pos=src,
                          grid_origin=obj_origin,
                          obj_size=objsize,
                          voxelsize=voxelsize,
                          det_corners=det_corners,
                          det_size=det_size,
                          pixel_sizeL=pixelsizeL,
                          pixel_sizeS=pixelsizeS,
                          fan_angle=fan,
                          E=E,
                          prob=prob,
                          rho=rho)
    
    tool.Emit()
    # tool.Scatter()

    #------------------ test emit ------------------#
    p_ptr = np.load("./data/emit_pptr.npy")
    m_idx = np.load("./data/emit_midx.npy")
    emit_data = np.load("./data/emit_data.npy")
    emit_vec = np.load("./data/emit_vec.npy")
    # 取出射线
    l, r = p_ptr[75], p_ptr[76]
    midx = m_idx[l:r]
    emitdata = emit_data[l:r]
    emitvec = emit_vec[l:r]
    tool.emit_data = {
        "p_ptr": np.array([0, r]),
        "m_idx": midx.astype(np.int32),
        "data": emitdata.astype(np.float64),
        "vec": emitvec.astype(np.float64),
        "shape": (1, r)
    }
    # i = np.floor(midx / 700)
    # j = midx % 700
    # plt.scatter(j, i)
    # plt.show()
    #------------------ test emit ------------------#
    
    tool.ScatterM()
    
    #------------------ test through_slit ------------------#
    # proj = tool.scatter_data
    # for i in range(350, 360):
    #     proj_i = proj[i].reshape((50, 50))
    #     print(np.max(proj_i), np.min(proj_i))
    #     plt.imshow(proj_i, cmap='gray', aspect='auto')
    #     plt.show()
    #------------------ test through_slit ------------------#
    
    #------------------ test sysmatrix ------------------#
    # decay = np.load("./data/scatter_decay_angle0.npy")  #[700*2500]
    # decay = decay.reshape(700, 2500)
    # data_angle0 = tool.scatter_data[4900:5600, :]  #[700, 2500]
    # tool.scatter_data[4900:5600, :] = decay * data_angle0
    # print(np.min(tool.scatter_data), np.max(tool.scatter_data))
    tool.Cal_SysMatrix()
    per = np.load("./sys_matrix/per000.npy")
    sys = np.load("./sys_matrix/decay000.npy")
    # for i in range(5250, 5260):
    #     proj_i = sys[i].reshape((50, 50))
    #     plt.imshow(proj_i, cmap='gray', aspect='auto')
    #     plt.show()
    z = tool.BackProjection(per, sys, detResponse)
    z_nodefect = tool.BackProjection(per, sys, detResponse_nodefect)
    # diff = z - z_nodefect
    # x = np.arange(diff.shape[0])
    # plt.plot(x, diff)
    # plt.show()
    #------------------ test sysmatrix ------------------#
    
    
    
    