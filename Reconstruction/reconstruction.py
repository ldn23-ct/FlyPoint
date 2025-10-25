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
    
    def Emit(self, save=False, Debug=0):
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
        
        # if save:
            # np.save("./data/temp_var/emit_pptr.npy", self.emit_data["p_ptr"])
            # np.save("./data/temp_var/emit_midx.npy", self.emit_data["m_idx"])
            # np.save("./data/temp_var/emit_data.npy", self.emit_data["data"])
            # np.save("./data/temp_var/emit_vec.npy", self.emit_data["vec"])
        
        if Debug:
            p_ptr = np.load("./data/temp_var/emit_pptr.npy")
            m_idx = np.load("./data/temp_var/emit_midx.npy")
            emit_vec = np.load("./data/temp_var/emit_vec.npy")
            # 取出射线
            # l, r = p_ptr[0], p_ptr[1]
            l, r = p_ptr[75], p_ptr[76]
            midx = m_idx[l:r]
            emitdata = np.ones(midx.shape[0])
            emitvec = emit_vec[l:r]
            tool.emit_data = {
                "p_ptr": np.array([0, r]),
                "m_idx": midx.astype(np.int32),
                "data": emitdata.astype(np.float64),
                "vec": emitvec.astype(np.float64),
                "shape": (1, int(l-r))
            }
            if Debug > 1:
                i = np.floor(midx / 700)
                j = midx % 700
                plt.scatter(j, i)
                plt.show()
 
    def ScatterM(self, Debug=0):
        # inl = InL.CalIntersectionLength(path="./Reconstruction/Vertex.yaml")
        # _, detS = DetArray(self.detcorners, self.pixelsizeS, self.detsize)
        # self.scatter_data = inl.through_slit(obj_array=self.obj_slice,
        #                                      det_array=detS,
        #                                      scale = int(self.pixelsizeL[0] / self.pixelsizeS[0]),
        #                                      threshold=1000,
        #                                      save=True)  # [m, n] [7e3, 2.5e3]
        self.scatter_data = np.load("./data/temp_var/scatter_data.npy")
        
        if Debug:
            length = self.scatter_data.reshape((self.obj_slice.shape[0], 50, 50))
        
        #------------------ calculate solid angle ------------------#
        det_normal, self.detL = DetArray(self.detcorners, self.pixelsizeL, self.detsize)
        x, y = np.arange(self.obj_slice.shape[0]), np.arange(self.detL.shape[0])
        m_idx, n_idx = np.meshgrid(x, y, indexing='ij')
        start = self.obj_slice[m_idx]
        end = self.detL[n_idx]
        vec = (end - start).reshape(-1, 3)
        r = np.linalg.norm(vec, axis=1)
        vec = vec / r[:, None]
        self.scatter_vec = vec.reshape((self.obj_slice.shape[0], self.detL.shape[0], 3))
        
        cos_phi = np.abs(np.clip(vec@det_normal, -1.0, 1.0))
        solid_angle = self.pixelsizeL[0]*self.pixelsizeL[1] * cos_phi / np.square(r)
        # 立体角很小，乘以系数因子，整体进行缩放
        scale = 1 / np.max(solid_angle)
        solid_angle = scale * solid_angle
        self.scatter_data = (solid_angle * self.scatter_data).reshape((self.obj_slice.shape[0], self.detL.shape[0]))
        if Debug:
            length_angle = self.scatter_data.reshape((self.obj_slice.shape[0], 50, 50))
            length1 = length[4950, :, :]
            length2 = length[5150, :, :]
            length3 = length[5350, :, :]
            length4 = length[5550, :, :]
            length_angle1 = length_angle[4950, :, :]
            length_angle2 = length_angle[5150, :, :]
            length_angle3 = length_angle[5350, :, :]
            length_angle4 = length_angle[5550, :, :]
            imgs = [length1, length2, length3, length4,
                    length_angle1, length_angle2, length_angle3, length_angle4]
            titles = ["5mm", "25mm", "45mm", "65mm"]
            fig, axes = plt.subplots(2, 4)
            for j, img in enumerate(imgs):
                axes[int(j/4), j%4].imshow(img, cmap='gray', aspect='auto')
                axes[int(j/4), j%4].set_title(titles[j%4])
            plt.tight_layout()
            plt.show()
        #------------------ calculate solid angle ------------------#
        
        #------------------ calculate decay ------------------#
        
        #------------------ calculate decay ------------------#
 
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
        dsdo = 0.5 * (k**2) * (k + invk - sin2)  # re_cm是常数，为了保证数值精度，整体进行缩放
        dsdo = np.sum(dsdo * prob[:, None], axis=0)
        return dsdo
        
    def Cal_SysMatrix(self, save=False, Debug=0):
        A_ptr = self.emit_data["p_ptr"]; Am_idx = self.emit_data["m_idx"]
        A_data = self.emit_data["data"]; A_vec = self.emit_data["vec"]
        P = A_ptr.shape[0] - 1
        M = self.scatter_data.shape[0]
        N = self.detL.shape[0]
        
        for p in tqdm(range(P)):
            sys_matrix_p = np.zeros((M, N))
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
            kn = kn / np.max(kn)
            coffei = coffei * kn  # [m_p, 2.5e3]
            sys = coffei / np.sum(coffei, axis=1)[:, None]
            sys_strengthen, _, _ = sigmoid_gate_double_anchor(sys.reshape((sys.shape[0], 50, 50)))
            sys_strengthen = sys_strengthen.reshape((sys.shape[0], -1))
            
            for i in range(coffei.shape[0]):
                sys_matrix_p[m_idx[i], :] = sys_strengthen[i, :]
            if Debug:
                if Debug > 1:
                    index = [int(m_idx.shape[0]/4), int(m_idx.shape[0]/2), int(m_idx.shape[0]*0.75)]
                    scatter_i = scatter_data[index].reshape((3, 50, 50))
                    coffei_i = coffei[index].reshape(3, 50, 50)
                    sys_i = sys[index].reshape(3, 50, 50)
                    sys_str_i = sys_strengthen[index].reshape(3, 50, 50)
                    fig, axes = plt.subplots(4, 3)
                    for j in range(3):
                        axes[0, j].imshow(scatter_i[j], cmap='gray', aspect='auto')
                        axes[1, j].imshow(coffei_i[j], cmap='gray', aspect='auto')
                        axes[2, j].imshow(sys_i[j], cmap='gray', aspect='auto')
                        axes[3, j].imshow(sys_str_i[j], cmap='gray', aspect='auto')
                    plt.show()
                
                    # proj_coffei = np.sum(coffei.reshape((coffei.shape[0], 50, 50)), axis=0)
                    # proj_sys = np.sum(sys.reshape((sys.shape[0], 50, 50)), axis=0)
                    # proj_sysstr = np.sum(sys_strengthen.reshape((sys_strengthen.shape[0], 50, 50)), axis=0)              
                    # fig, axes = plt.subplots(1, 3)
                    # axes[0].imshow(proj_coffei, cmap='gray', aspect='auto')
                    # axes[1].imshow(proj_sys, cmap='gray', aspect='auto')
                    # axes[2].imshow(proj_sysstr, cmap='gray', aspect='auto')
                    # plt.show()
                
                if Debug > 2:
                    temp = np.zeros((M, N))
                    for i in range(coffei.shape[0]):
                        temp[m_idx[i], :] = sys[i, :]
                    return sys_matrix_p, temp
                
                return sys_matrix_p
               
            if save:
                np.save(f"./sys_matrix/sys{p:03d}.npy", sys_matrix_p)
            
    def BackProjection(self, sys_matrix: np.ndarray, det_response: np.ndarray, Debug=0):
        '''
        sys_matrix: shape [m, n]
        det_response: shape [n,]
        '''
        vox = np.sum(sys_matrix * det_response[None, :], axis=1)
        proj = vox.reshape((int(vox.shape[0]/self.nz), self.nz))
        if Debug:
            # plt.imshow(proj, cmap="gray", aspect='auto')
            z = np.sum(proj, axis=0)
            x =  np.arange(z.shape[0])
            plt.plot(x, z)
            plt.show()
            return vox, z
        return vox
            
def sigmoid_gate_double_anchor(
    A: np.ndarray,
    r: float = 0.05,
    p1: float = 70.0,
    p2: float = 95.0,
    w1: float = 0.10,
    w2: float = 0.90,
    eps_rel: float = 1e-6,   # 防零相对量
    eps_abs: float = 1e-30   # 防零绝对量
):
    """
    r : 非主通道的最小残留权重，区间 [0,1)
    p1, p2 : 对数域分位点（百分位，按每个 i 独立计算）
    w1, w2 : 在对应分位点处的目标权重，均应在 (r,1) 内
    eps_rel, eps_abs : 计算 log(B+eps) 的防零项

    A_out : [M, N, P]，平滑加权后的矩阵，且每块 ∑ 等于修改前对应块的 ∑（能量守恒）
    Wcol  : [M, P]，列权重（用于诊断/可视化）
    params: (tau_L, T) 各为 [M,1]，分别是对数域的阈值与温度（每块一组）
    """
    M, N, P = A.shape

    # 原块和（能量回标定用）
    S0 = A.sum(axis=(1,2), keepdims=True)  # [M,1,1]

    # 列强度与对数
    B = A.sum(axis=1)                       # [M,P]
    # 每块独立 eps：与该块最大列强度同量级，避免 log(0)
    eps = np.maximum(B.max(axis=1, keepdims=True) * eps_rel, eps_abs)
    L = np.log(B + eps)                     # [M,P]

    # 分位函数（按块）
    def q(x: np.ndarray, pct: float) -> np.ndarray:
        return np.percentile(x, pct, axis=1, keepdims=True)  # [M,1]

    L1 = q(L, p1)   # [M,1]
    L2 = q(L, p2)   # [M,1]

    # 目标权重 -> logistic 空间
    def to_g(w: float, r: float) -> float:
        z = (w - r) / (1.0 - r)
        z = np.clip(z, 1e-6, 1.0 - 1e-6)   # 稳定化
        return float(np.log(z / (1.0 - z)))

    g1 = to_g(w1, r)
    g2 = to_g(w2, r)

    # 反推出每块的 T 与 tau_L
    denom = (g2 - g1)
    if abs(denom) < 1e-12:
        denom = np.sign(denom) * 1e-12 + (denom == 0) * 1e-12

    T = (L2 - L1) / denom        # [M,1]
    # 数值下限，防止极小值导致过陡（可按需调小/关闭）
    T = np.where(T <= 1e-12, 1e-12, T)

    tau_L = L1 - T * g1          # [M,1]

    # 列权重（平滑 sigmoid，取值 [r,1]）
    Wcol = r + (1.0 - r) / (1.0 + np.exp(-(L - tau_L) / T))  # [M,P]

    # 应用到 A 并能量回标定（保证每块总和回到原值 S0）
    A_tmp = A * Wcol[:, None, :]                  # [M,N,P]
    S = A_tmp.sum(axis=(1,2), keepdims=True)      # [M,1,1]
    A_out = A_tmp.copy()
    mask = (S > 0)
    ratio = S0[mask] / S[mask]
    A_out[mask.squeeze()] *= ratio[:, None, None]

    return A_out, Wcol, (tau_L, T)


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
    voxelshape = (objsize / voxelsize).astype(np.int32)
    rho = np.zeros(voxelshape)
    rho.fill(0.134*2.7*1e-3)
    
    detResponses = np.load("./data/MC_data/0_degree_interval.npy")[:, ::-1]
    detResponses_nodefect = np.load("./data/MC_data//0_degree_no_defect.npy")[:, ::-1]
    # plt.imshow(temp, cmap="gray", aspect='auto')
    # plt.scatter(np.arange(detResponses.shape[0]), np.sum(detResponses, axis=0))
    # plt.scatter(np.arange(detResponses_nodefect.shape[0]), np.sum(detResponses_nodefect, axis=0))
    # plt.show()
    
    detResponse = detResponses.ravel()
    detResponse_nodefect = detResponses_nodefect.ravel()
    res = detResponse / detResponse_nodefect - 1
    
    
    
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
    
    tool.Emit(Debug=1)
    
    tool.ScatterM(Debug=0)
    
    #------------------ test sysmatrix ------------------#
    sys = tool.Cal_SysMatrix(save=False, Debug=1)
        
    dbug = 1
    vox_res, z = tool.BackProjection(sys, res, Debug=dbug) 
    vox_defect, z_defect = tool.BackProjection(sys, detResponse, Debug=dbug)
    vox_nodefect, z_nodefect = tool.BackProjection(sys, detResponse_nodefect, Debug=dbug)
    
    if dbug > 1:
        _, sys_orig = tool.Cal_SysMatrix(save=False, Debug=3)
        vox_res_orig, z_orig = tool.BackProjection(sys_orig, res, Debug=dbug) 
        vox_defect_orig, z_defect_orig = tool.BackProjection(sys_orig, detResponse, Debug=dbug)
        vox_nodefect_orig, z_nodefect_orig = tool.BackProjection(sys_orig, detResponse_nodefect, Debug=dbug)
        
        fig, axes = plt.subplots(2, 3)
        col_titles = ["res", "defect", "nodefect"]
        row_titles = ["processed", "original"]
        values = [z, z_defect, z_nodefect, z_orig, z_defect_orig, z_nodefect_orig]
        x = np.arange(z.shape[0])
        
        for i, img in enumerate(values):
            axes[int(i/3), i%3].plot(x, img)
        for ax, col in zip(axes[0], col_titles):
            ax.set_title(col, fontsize=12)
        for ax, row in zip(axes[:,0], row_titles):
            ax.set_ylabel(row, labelpad=15, rotation=90, ha='right', va='center')
        plt.tight_layout()
        plt.show()    
    #------------------ test sysmatrix ------------------#
    
    #------------------ test spatial resolution ------------------#
    detResponse_5mm = (np.load("./data/MC_data/0_degree_interval_5mm.npy")[:, ::-1]).ravel()
    detResponse_3mm = (np.load("./data/MC_data/0_degree_interval_3mm.npy")[:, ::-1]).ravel()
    res_5mm = detResponse_5mm / detResponse_nodefect - 1
    res_3mm = detResponse_3mm / detResponse_nodefect - 1
    
    _, z = tool.BackProjection(sys, res_5mm, Debug=dbug) 
    _, z_defect = tool.BackProjection(sys, detResponse_5mm, Debug=dbug)
    _, z = tool.BackProjection(sys, res_3mm, Debug=dbug) 
    _, z_defect = tool.BackProjection(sys, detResponse_3mm, Debug=dbug)
    #------------------ test spatial resolution ------------------#
    