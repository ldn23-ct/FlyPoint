import numpy as np
import incident as Inc

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
        corner_pos--4 corners position of the detector
        dx/dy--length of the detector pixel
        x/y--length of the detector
    output
        detarray--pos of the pixels' center shape:[ny, nx]
    idea
        (1) 计算像素个数
        (2) 计算每个像素中心点
    '''
    nx, ny = int(detsize[0] / pixelsize[0]), int(detsize[1] / pixelsize[1])
    P0, P1, P2, P3 = map(np.array, corners)
    centers = []

    for i in range(ny):
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
    return np.array(centers)

class HalfSpaceCutting:
    '''
    判断射线束是否与立方体相交，同时输出相交长度 \\
    (1) 根据平面角点，构造有序顶点集合并计算向外法向量 \\
    (2) 读取空间点集合，构造 {pi, qi} 矩阵 \\
    (3) 验证线段是否与平面相交，若相交求出参数t \\
    (4) 对六面取交集，判断是否经过狭缝，输出mask \\
    (5) 对于经过狭缝的射线，计算出射衰减
    '''
    def plane_normal(self, verts: np.ndarray, centroid: np.ndarray):
        '''
        input
            verts: 有序顶点集合，按逆时针排序 shape:[4, 3], 左上角开始
            centroid: 六面体几何中心 shape:[3,]
        output
            normal: 向外归一化法向量 shape:[3,]
            m: 面中心点 shape:[3,]
        '''
        v0, v1, v2 = verts[0], verts[1], verts[2]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("collinear vertices")
        m = verts.mean(axis=0)
        if np.dot(n, m - centroid) < 0: 
            verts = verts[::-1].copy()
            n = -n
        return verts, n / nn, m

    def loda_point(self, obj_array, det_array, r_array):
        '''
        input
            obj_array--shape:[a*b, 3]
            det_array--shape:[c*c, 3]
            r_array--shape:[6, 3]
        output
            p--(det_array - obj_array) shape:[a*b, c*c, 3]
            q--(obj_array - m) shape:[6, a*b, 3]
        '''
        p = det_array[None, :, :] - obj_array[:, None, :]
        q = obj_array[None, :, :] - r_array[:, None, :]
        return p, q
    
    def build_six_faces(self, entrance, exit):
        '''
        input
            entrance--shape:[4, 3]
            exit--shape:[4, 3]
        output
            faces--[dics{name, verts, n, r}]
        '''
        corners = np.vstack((entrance, exit))
        centroid = np.mean(corners, axis=0)
        # faces = []
        A4, nA, rA = self.plane_normal(entrance, centroid)
        B4, nB, rB = self.plane_normal(exit, centroid)
        ns, rs = [nA, nB], [rA, rB]
        # faces.append({"name": "A", "verts": A4, "n": nA, "r": rA})
        # faces.append({"name": "B", "verts": B4, "n": nB, "r": rB})
        for k in range(4):
            verts = np.array([exit[k], entrance[k], entrance[(k+3)%4], exit[(k+3)%4]])
            k4, nk, rk = self.plane_normal(verts, centroid)
            # faces.append({"name": str(k), "verts": k4, "n": nk, "r": rk})
            ns.append(nk)
            rs.append(rk)
        return np.array(ns), np.array(rs)
    
    def through_slit(self, p: np.ndarray, q: np.ndarray, ns: np.ndarray, require_enter=True, eps_base=1e-9, enter_plane=0, exit_plane=1):
        '''
        input
            p--(det_array - obj_array) shape:[a*b, c*c, 3]
            q--(obj_array - r) shape:[6, a*b, 3]
            ns--shape:[6, 3]
        output
            csm: {'m_idx','n_idx','vec','ptr'}
        idea
            线段穿过狭缝的判据
            1. 不存在外侧平行面
            2. 进入面最大值与离开面最小值分别在A/B取得
            3. 进入面最大值要小于离开面最小值        
        '''
        m, n, _ = p.shape
        assert q.shape == (6, m, 3)
        
        scale = max(1.0,
                    float(np.max(np.linalg.norm(p.reshape(-1,3), axis=1))) if p.size else 1.0,
                    float(np.max(np.linalg.norm(q.reshape(-1,3), axis=1))) if q.size else 1.0,
                    float(np.max(np.linalg.norm(ns, axis=1))) if ns.size else 1.0)
        eps = eps_base * scale
        
        tE = np.full((m, n), -np.inf, dtype=p.dtype)
        tL = np.full((m, n), np.inf, dtype=p.dtype)
        enter_idx = np.full((m, n), -1, dtype=np.int32)
        exit_idx = np.full((m, n), -1, dtype=np.int32)
        outside_para_mask = np.zeros((m, n), dtype=bool)
        
        for k in range(ns.shape[0]):
            n_k, qs_k = ns[k], q[k, ...]
            ps = - (p @ n_k)    # [m, n]
            qs = (qs_k @ n_k) [:, None]     # [m, 1]
            
            enter_mask = ps > eps
            exit_mask = ps < -eps
            para = np.abs(ps) <= eps
            outside_para_mask |= para & (qs > eps)
            
            t = np.zeros_like(ps)
            np.divide(qs, ps, out=t, where=~para)
            
            better_enter_mask = enter_mask & (t > tE)
            if np.any(better_enter_mask):
                tE[better_enter_mask] = t[better_enter_mask]
                enter_idx[better_enter_mask] = k
            
            better_exit_mask = exit_mask & (t < tL)
            if np.any(better_exit_mask):
                tL[better_exit_mask] = t[better_exit_mask]
                exit_idx[better_exit_mask] = k
        
        t0 = np.maximum(tE, 0)
        t1 = np.minimum(tL, 1)

        valid = (~outside_para_mask) & (tE <= tL + eps) & (t0 <= t1 + eps)

        if require_enter:
            valid = valid & (enter_idx == enter_plane) & (exit_idx == exit_plane)
        
        m_idx, n_idx = np.nonzero(valid)
        order = np.argsort(m_idx, kind='stable')
        m_idx = m_idx[order]; n_idx = n_idx[order]
        v = p[m_idx, n_idx, :].astype(np.float64, copy=False)
        v = v / np.linalg.norm(v, axis=1)[:, None]     
        m_unique, counts = np.unique(m_idx, return_counts=True)
        
        m_cnts = np.zeros(m); ptr = np.zeros(m+1); sum = 0
        for k in range(m_unique.shape[0]):
            m_cnts[m_unique[k]] = counts[k]
        for k in range(1, m+1):
            sum += m_cnts[k-1]
            ptr[k] = sum
        
        # return v.astype(np.float32, copy=False), ptr, n_idx.astype(np.int32, copy=False)
        return {
            "m_ptr": ptr,
            "vec": v.astype(np.float32, copy=False),
            "m_idx": m_idx.astype(np.int32, copy=False),
            "n_idx": n_idx.astype(np.int32, copy=False),
            "shape": (m, n)
        }
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
                 pixel_size: np.ndarray,  #shape: [2,]
                 fan_angle: np.float32,
                 slit_corners: list,  # shape: [8, 3]
                 E: np.ndarray,  # shape: [e,]
                 prob: np.ndarray,  # shape: [e,]
                 rho: np.ndarray  # shape: ?
                 ):
        self.src = src_pos
        self.obj_origin = grid_origin
        self.objsize = obj_size
        self.voxelsize = voxelsize
        self.detsize = det_size
        self.pixelsize = pixel_size
        self.det = DetArray(det_corners, self.pixelsize, self.detsize)  # 按行排序，右下角为起点
        self.fan = np.deg2rad(fan_angle)
        self.slit = slit_corners
        self.E = E
        self.prob = prob / np.sum(prob)
        self.rho = rho
    
    def Emit(self):
        SOD = self.obj_origin[2] - self.src[2]
        slice_halfy = (SOD + self.objsize[2]) * np.tan(self.fan / 2)
        ny = int(2 * np.ceil(slice_halfy / self.voxelsize[1]))
        nz = int(np.ceil(self.objsize[2] / self.voxelsize[2]))
        obj_slice_size = [ny * self.voxelsize[1], self.objsize[2]]
        #------------------ slice sample ------------------#
        y_start = (obj_slice_size[0] - self.voxelsize[1]) / 2
        z_start = SOD + self.voxelsize[2] / 2
        y_centers = [(y_start - i * self.voxelsize[1]) for i in range(ny)]
        z_centers = [(z_start + i * self.voxelsize[2]) for i in range(nz)]
        Y, Z = np.meshgrid(y_centers, z_centers, indexing='ij')
        X = np.zeros_like(Y)
        self.obj_slice = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # 按行排序, 左上角为起点
        #------------------ slice sample ------------------#
        self.emit_data = Inc.incident_vector_calulate(SOD,
                                                      obj_slice_size,
                                                      ny,
                                                      nz,
                                                      self.fan, 
                                                      ray_step=np.deg2rad(1),
                                                      voxels_size=self.voxelsize)

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
        objcorner = self.objsize / 2
        objcorner[2] = self.obj_origin[2] - self.src[2]
        start_point = self.obj_slice[self.scatter_data["m_idx"]]
        end_point = self.det[self.scatter_data["n_idx"]]
        decay = Inc.voxel_path_length_cal(np.array([-100,-100,60]), self.voxelsize, self.objsize, start_point, end_point)
        assert solid_angle.shape[0] == decay.shape[0]
        # decay = 1
        self.scatter_data["data"] = solid_angle * decay
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
        dsdo = 0.5 * (re_cm**2) * (k**2) * (k + invk - sin2)  # shape: [e, n]
        dsdo = np.sum(dsdo * prob[:, None], axis=0)
        return dsdo * 100
        
    def Cal_SysMatrix(self):
        #------------------ find nozero-vectors ------------------#
        A_ptr = self.emit_data["m_ptr"]; Ap_idx = self.emit_data["p_idx"]
        A_data = self.emit_data["data"]; A_vec = self.emit_data["vec"]
        B_ptr = self.scatter_data["m_ptr"]; Bn_idx = self.scatter_data["n_idx"]; 
        B_vec = self.scatter_data["vec"]; B_data = self.scatter_data["data"]
        A_shape = self.emit_data["shape"]; B_shape = self.scatter_data["shape"]
        M = A_shape[1]
        assert B_shape[0] == M
        
        cos_theta, m_idx, p_idx, n_idx, coeffi = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        m_ptr = np.zeros(M + 1)
        for m in range(M):
            a0, a1 = A_ptr[m], A_ptr[m+1]
            b0, b1 = B_ptr[m], B_ptr[m+1]
            if a0 == a1 or b0 == b1: continue
            p_sub, n_sub = Ap_idx[a0:a1], Bn_idx[a0:a1]
            emit_vec, emit_data = A_vec[a0:a1], A_data[a0:a1]
            scatter_vec, scatter_data = B_vec[b0:b1], B_data[b0:b1]
            
            #---------------kn项余弦---------------#
            costheta = (emit_vec @ scatter_vec.T).ravel()  # shape: [a*b,]
            np.clip(costheta, -1.0, 1.0, out=costheta)
            cos_theta = np.hstack(cos_theta, costheta)
            #---------------kn项余弦---------------#
            
            #---------------入/出射衰减及立体角---------------#
            k = (emit_data @ scatter_data.T).ravel()  # shape: [a*b,]
            coeffi = np.hstack(coeffi, k)
            #---------------入/出射衰减及立体角---------------#
            
            #---------------索引网格重组---------------#
            # m_ptr[m + 1] = costheta.shape[0]
            ps, ns = np.meshgrid(p_sub, n_sub)
            m_idx = np.hstack(m_idx, np.array([m]*ps.shape[0]))
            p_idx = np.hstack(p_idx, ps.ravel())
            n_idx = np.hstack(n_idx, ns.ravel())
            #---------------索引网格重组---------------#
        
        # kn项，需要结合物体电子密度计算
        kn = self.klein_nishina(cos_theta)
        coeffi = coeffi * kn
        
        # 不同p之间进行合并
        sys_matrix = np.zeros(B_shape)
        for i in range(coeffi.shape[0]):
            sys_matrix[m_idx[i], n_idx[i]] += coeffi[i]
            
    def BackProjection():
        pass
            
            

if __name__ == "__main__":
    src = np.array([0, 0, 0])
    obj_origin = np.array([0, 0, 60])
    objsize = np.array([200, 200, 70])
    fan = 12
    voxelsize = np.array([5, 5, 0.1])
    det_size = np.array([50, 50])
    pixelsize = np.array([1, 1])
    det_corners = [[60, -25, 40], [60, 25, 40], [110, 25, 40], [110, -25, 40]]
    # slitcorners = np.array([[56.31, 25, 66.32], [56.31, -25, 66.32], [55.54, -25, 66.97], [55.54, 25, 66.97],
    #                    [60.16, 25, 70.92], [60.16, -25, 70.92], [59.4, -25, 71.56], [59.4, 25, 71.56]])
    slitcorners = np.array([[54, 25, 60], [54, -25, 60], [53, -25, 60], [53, 25, 60],
                            [54, 25, 59], [54, -25, 59], [53, -25, 59], [53, 25, 59]])
    E = np.array([120, 160])
    prob = np.array([0.4, 0.6])
    rho = 1
    
    tool = ReConstruction(src,
                          obj_origin,
                          objsize,
                          voxelsize,
                          det_corners,
                          det_size,
                          pixelsize,
                          fan,
                          slitcorners,
                          E,
                          prob,
                          rho)
    
    tool.Emit()
    tool.Scatter()
    print("done")
    

