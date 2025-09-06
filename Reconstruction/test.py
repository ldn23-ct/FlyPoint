import numpy as np

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
            # tr = v_top_start + (v_top_end - v_top_start) * (j + 1) / nx  # top-right
            br = v_bot_start + (v_bot_end - v_bot_start) * (j + 1) / nx  # bottom-right
            # bl = v_bot_start + (v_bot_end - v_bot_start) * j / nx        # bottom-left

            center = (tl + br) / 2  # diagonal midpoint
            centers.append(center)

    centers = np.array(centers).reshape(ny, nx)

    return centers

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
    
    def through_slit(self, p, q, ns, require_enter=True, eps_base=1e-9):
        '''
        input
          p--(det_array - obj_array) shape:[a*b, c*c, 3]
          q--(obj_array - r) shape:[6, a*b, 3]
          ns--shape:[6, 3]
        output
          t0--enter point, shape:[a*b, c*c]
          t1--exit point, shape:[a*b, c*c]
          valid--through_slit_ray, shape:[a*b, c*c]
        idea
          线段穿过狭缝的判据
          1. 不存在外侧平行面
          2. 进入面最大值与离开面最小值分别在A/B取得
          3. 进入面最大值要小于离开面最小值        
        '''
        scale = max(1.0,
                    float(np.max(np.linalg.norm(p.reshape(-1,3), axis=1))) if p.size else 1.0,
                    float(np.max(np.linalg.norm(q.reshape(-1,3), axis=1))) if q.size else 1.0,
                    float(np.max(np.linalg.norm(ns, axis=1))) if ns.size else 1.0)
        eps = eps_base * scale
        ps = -np.einsum('ij,drj->idr', ns, p)  # [6, a*b, c*c]
        qs = np.einsum('ij,idj->id', ns, q)[:, :, None]  # [6, a*b, 1]
        ts = qs / ps  # [6, a*b, c*c]

        enter_mask = ps > eps  # [6, a*b, c*c]
        exit_mask = ps < -eps  # [6, a*b, c*c]
        outside_para_mask = (~(enter_mask | exit_mask)) & (qs > eps)
        outside_para_mask = np.any(outside_para_mask, axis=0)  # [a*b, c*c]

        neg_inf, pos_inf = -np.inf, np.inf
        t_enter = np.where(enter_mask, ts, neg_inf)  # [6, a*b, c*c]
        tE = np.max(t_enter, axis=0)  # [a*b, c*c]
        enter_idx = np.argmax(t_enter, axis=0)
        # print(enter_idx[0].reshape(7, 7))
        
        t_leave = np.where(exit_mask, ts, pos_inf)  # [6, a*b, c*c]
        tL = np.min(t_leave, axis=0)  # [a*b, c*c]
        exit_idx = np.argmin(t_leave, axis=0)
        # print(exit_idx[0].reshape(7, 7))

        t0 = np.maximum(tE, 0)
        t1 = np.minimum(tL, 1)

        valid = (~outside_para_mask) & (tE < tL + eps) & (t0 < t1 + eps)
        # print(valid[0].reshape(7, 7))
        if require_enter:
          valid = valid & (enter_idx == 0) & (exit_idx == 1)

        return t0, t1, valid

class ScatterVec:
    '''
    在半裁剪算法的基础上, 计算出射衰减, 以及是否穿过狭缝 \\
    input: \\
      objcorners--模体角点, 需要计算射线路径, ndarray of shape: [8, 3] \\
      slitcorners--狭缝角点, 需要计算是否穿过狭缝, ndarray of shape: [8, 3] \\
      obj--模体采样点, ndarray of shape: [m, 3] \\
      det--探测器采样点, ndarray of shape: [n, 3] \\
    output: \\
      ScatterVector--散射射线向量, 被遮挡记作nan, 否则方向表示向量, 模长表示衰减, ndarray of shape: [a*b, c*c] \\
      Angle--与探测器夹角余弦值, 用于计算kn项
    '''
    def __init__(self, objcorners, slitcorners, dDet):
        self.hc = HalfSpaceCutting()
        self.ns_obj, self.rs_obj = self.hc.build_six_faces(objcorners[0:4], objcorners[4:])
        self.ns_slit, self.rs_slit = self.hc.build_six_faces(slitcorners[0:4], slitcorners[4:])
        self.dA = dDet

    def SlitCalculate(self):
        '''
        计算是否遮挡, 同时计算探测器面元对体素点的立体角 \\
        探测器平面与狭缝平行
        '''
        p, q = self.hc.loda_point(obj, det, self.rs_slit)
        _, _, valid = self.hc.through_slit(p, q, self.ns_slit)
        I, _ = np.nonzero(valid)
        p_valid = p[I]
        temp = p_valid - np.dot(p_valid, self.ns_slit[0])*self.ns_slit[0]
        cos_phi = np.linalg.norm(temp) / np.linalg.norm(p_valid)
        solid_angle = self.dA * cos_phi / (np.linalg.norm(p_valid))**2
        return cos_phi, I


    def SVCalculate(self, obj, det):
        p, q = self.hc.loda_point(obj, det, self.rs_obj)
        _, valididx = self.SlitCalculate()
        t0, t1, _ = self.hc.through_slit(p, q, self.ns_obj)
        pathLength = p[valididx] * (t1[valididx] - t0[valididx])
        return pathLength



if __name__ == "__main__":
    A4 = [[-1, 1, -1], [-1, -1, -1], [1, -1, -1], [1, 1, -1]]
    B4 = [[-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1]]
    func = HalfSpaceCutting()
    ns, rs = func.build_six_faces(np.array(A4), np.array(B4))
    print(ns)
    print(rs)
    
    # x1, x2 = np.arange(-3, 4, 3), np.arange(-3, 4)
    # y1, y2 = np.arange(-3, 4, 3), np.arange(-3, 4)
    # X1, Y1 = np.meshgrid(x1, y1)
    # X2, Y2 = np.meshgrid(x2, y2)
    # Z1, Z2 = -5*np.ones(X1.shape), 5*np.ones(X2.shape)
    # obj = np.column_stack((X1.ravel(), Y1.ravel(), Z1.ravel())).astype(np.float64, copy=False)
    # det = np.column_stack((X2.ravel(), Y2.ravel(), Z2.ravel())).astype(np.float64, copy=False)

    obj = np.array([[-3, -3, -5]])
    det = np.array([[3, 2, 5]])

    p, q = func.loda_point(obj, det, rs)
    t0, t1, valid = func.through_slit(p, q, ns)
    # print(obj[4, :])
    print(valid.reshape(p.shape[0], p.shape[1]))
    # print(t0[0].reshape(x2.shape[0], y2.shape[0]))
    # print(t1[0].reshape(x2.shape[0], y2.shape[0]))
    # print(valid[0].reshape(x2.shape[0], y2.shape[0]))

