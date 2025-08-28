import numpy as np

def Mapping(pos: np.ndarray, voxelsize: np.ndarray, kernalsize: np.ndarray):
    '''
    input
      pos--模体前表面左上角点坐标 初始时刻对准零点 ndarray of shape (2,) dtype=float
      dx/dy/dz--空间体素大小 根据空间坐标计算体素编号
      kernal_y/kernal_z--二维卷积核尺寸 大小与体素大小相同 计算会极大简化
    output
      map--二维卷积核每个像素与三维空间体素的对应关系
    idea
      (1) 空间体素划分编号 n = iz + iy * nz + ix * (ny + nz)
      (2) 根据 pos_x 找到 ix
      (3) 假设零点是对准的 kernal相当于三维矩阵的一个切片 只需要根据 pos_y 就可以求出 iy0
          并顺便取出每一行对应的 iy 
      (4) 当 pos_x 位于交界线上 归 x- 像素
      (5) 当 iy 越界 map赋值null
    '''
    ix = -pos[0] // voxelsize[0]

    return

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
    nx, ny = detsize[0] / pixelsize[0], detsize[1] / pixelsize[1]
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

def ObjArray(angle, fdis, bdis, kernalsize):
  '''
  input
    angle--the angle of the fan beam
    fdis--the distance from source to the front surface of the object
    bdis--the distance from source to the back surface of the object
    kernel_y/kernel_z--size of kernal, equal to voxel
  output
    objarray--pos of the voxels' center
  '''
  angle = np.deg2rad(angle)
  height = bdis * np.tan(angle/2)
  ny = np.ceil(height / kernalsize[0])
  nz = np.ceil((bdis - fdis) / kernalsize[1])
  centers = np.zeros((ny*nz, 2))
  y0, z0 = height - kernalsize[0]/2, fdis + kernalsize[1]/2
  for i in range(ny):
      for j in range(nz):
          idx = j + i * nz
          centers[idx] = [y0 - i * kernalsize[0], z0 + j * kernalsize[1]]
  return centers

class HalfSpaceCutting:
    '''
    判断射线束是否与立方体相交，同时输出相交长度
    (1) 根据平面角点，构造有序顶点集合并计算向外法向量
    (2) 读取空间点集合，构造 {pi, qi} 矩阵
    (3) 验证线段是否与平面相交，若相交求出参数t
    (4) 对六面取交集，判断是否经过狭缝，输出mask
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

    def loda_point(self, obj_array, det_array, m_array):
        '''
        input
          obj_array--shape:[a*b, 3]
          det_array--shape:[c*c, 3]
          m_array--shape:[6, 3]
        output
          p--(det_array - obj_array) shape:[a*b, c*c, 3]
          q--(obj_array - m) shape:[6, a*b, 3]
        '''
        p = det_array[None, :, :] - obj_array[:, None, :]
        q = obj_array[None, :, :] - m_array[:, None, :]
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
        
        t_leave = np.where(exit_mask, ts, pos_inf)  # [6, a*b, c*c]
        tL = np.min(t_leave, axis=0)  # [a*b, c*c]
        exit_idx = np.argmin(t_leave, axis=0)

        t0 = np.maximum(tE, 0)
        t1 = np.minimum(tL, 1)

        valid = (~outside_para_mask) & (tE < tL + eps) & (t0 < t1 + eps)
        if require_enter:
          valid = valid & (enter_idx == 0) & (exit_idx == 1)

        return t0, t1, valid

if __name__ == "__main__":
    A4 = [[-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1]]
    B4 = [[-1, 1, -1], [-1, -1, -1], [1, -1, -1], [1, 1, -1]]
    func = HalfSpaceCutting()
    faces = func.build_six_faces(np.array(A4), np.array(B4))
    for i in range(6):
        face = faces[i]
        print(face["name"])
        print(face["verts"])
        print(face["n"])
        print(face["r"])

