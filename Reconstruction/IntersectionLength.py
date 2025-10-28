import numpy as np
import incident as Inc
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

HEX_FACE_IDX = {
    "bottom": [0,1,2,3],
    "top":    [4,5,6,7],
    "front":  [0,1,5,4],
    "right":  [1,2,6,5],
    "back":   [2,3,7,6],
    "left":   [3,0,4,7],
}       
        
class CalIntersectionLength:
    '''
    构建几何体并求穿过几何体路径长度
    '''
    def __init__(self, path):
        self.load_yaml(path)
        self.build_six_faces()
    
    def load_yaml(self, path: str) -> dict:
        '''
        加载挡板八顶点及衰减系数
        '''
        vertices = []
        mus = []
        rhos = []
        self.cube_num = 0
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for cube in data["bodies"]:
            self.cube_num += 1
            mus.append(cube["mu"])
            rhos.append(cube["rho"])
            vertices.append(cube["vertices"])
        self.vertices = np.array(vertices)      # [n, 8, 3]
        self.mus = np.array(mus) * np.array(rhos) * 1e-1   # [n,]
    
    def plane_norm(self, verts, idx, center):
        '''
        返回平面外法向及中心点
        '''
        v0, v1, v2, v3 = verts[idx[0]], verts[idx[1]], verts[idx[2]], verts[idx[3]]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("collinear vertices")
        c = (v0 + v1 + v2 + v3) / 4
        if np.dot(n, c - center) < 0:
            n = -n
        return n/nn, c
    
    def build_six_faces(self):
        '''
        构建每个挡板的法向量以及中心点
        '''
        ns, cs = [], []
        for i in range(self.cube_num):
            n, c = [], []
            cube_vertices = self.vertices[i]    # [8, 3]
            center = np.sum(cube_vertices, axis=0) / 8
            for _, idx in HEX_FACE_IDX.items():
                ni, ci = self.plane_norm(cube_vertices, idx, center)
                n.append(ni)
                c.append(ci)
            ns.append(n)
            cs.append(c)
        self.ns = np.array(ns)
        self.cs = np.array(cs)

    def calLength(self, src: np.ndarray, ends: np.ndarray, normals: np.ndarray, centers: np.ndarray, eps_base=1e-9):
        '''
        计算向量穿过挡板的距离并计算衰减
        '''
        # ---- dtype 与基本量 ----
        src  = np.asarray(src,     dtype=np.float64)
        ends = np.asarray(ends,    dtype=np.float64)
        nrm  = np.asarray(normals, dtype=np.float64)
        ctr  = np.asarray(centers, dtype=np.float64)
        eps  = float(eps_base)

        n = ends.shape[0]
        k = nrm.shape[0]
        if n == 0 or k == 0:
            return np.zeros((k, n), dtype=np.float64)

        # 方向与长度
        d = ends - src[None, :]                  # (n,3)
        seg_len = np.linalg.norm(d, axis=1)      # (n,)
        zero_dir = seg_len <= eps                # 零/近零长度线段直接置0

        # a = n·d  -> (n,k,6) ;  b = n·(c - s) -> (k,6) (与射线无关)
        a = np.einsum('nj,kij->nki', d, nrm)     # (n,k,6)
        b = np.sum(nrm * (ctr - src[None, None, :]), axis=2)  # (k,6)
        b_broad = b[None, :, :]                  # (n,k,6) for broadcasting

        # ---- 自适应阈值（随量纲缩放）----
        # |a| 阈值随 ||n||*||d|| 缩放； b 阈值随 ||n|| 缩放
        n_norm = np.linalg.norm(nrm, axis=2)         # (k,6)
        d_norm = np.maximum(seg_len, eps)            # (n,) 避免零
        eps_a = eps * (d_norm[:, None, None] * n_norm[None, :, :] + 1.0)  # (n,k,6)
        eps_b = eps * (n_norm + 1.0)                 # (k,6)

        # 并行/非并行
        parallel = np.abs(a) <= eps_a                # (n,k,6)
        nonpar   = ~parallel
        pos = (a >  eps_a) & nonpar                  # 上界：t <= b/a
        neg = (a < -eps_a) & nonpar                  # 下界：t >= b/a

        # 并行且在外侧：整条线段对该挡板无交（注意容差）
        invalid_parallel = parallel & (b_broad < -eps_b[None, :, :])  # (n,k,6)
        kill = invalid_parallel.any(axis=2)          # (n,k)

        # 比值 b/a；只在掩码处有意义
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = b_broad / a                      # (n,k,6)

        # 汇总上下界（不在这里 clamp [0,1]，最后一步再做）
        cand_hi = np.where(pos, ratio, np.inf)       # (n,k,6)
        cand_lo = np.where(neg, ratio, -np.inf)      # (n,k,6)
        t_hi = np.min(cand_hi, axis=2)               # (n,k)
        t_lo = np.max(cand_lo, axis=2)               # (n,k)

        # 被并行外侧判死的线段：强制无交
        t_lo[kill] = 1.0
        t_hi[kill] = 0.0

        # 端点掠触松弛
        eps_t = 10.0 * eps
        t_lo_relaxed = t_lo - eps_t
        t_hi_relaxed = t_hi + eps_t

        # 与线段域 [0,1] 求交，并转为物理长度
        t0 = np.maximum(0.0, t_lo_relaxed)          # (n,k)
        t1 = np.minimum(1.0, t_hi_relaxed)          # (n,k)

        inside = (t1 > t0 + eps_t)
        lengths_nk = np.zeros((n, k), dtype=np.float64)

        # 先整体计算再掩码置零，避免花哨布尔索引
        raw = (t1 - t0) * seg_len[:, None]          # (n,k)
        raw[~inside] = 0.0
        raw[zero_dir, :] = 0.0
        lengths_nk = raw

        # 需要 [k,n]
        return lengths_nk.T# (k,n)

    def through_slit(self, obj_array: np.ndarray, det_array: np.ndarray, scale: int, threshold, save=False):
        '''
        计算每个体素点到探测器点的狭缝穿过情况
        obj_array: [m, 3]
        det_array: [n, 3]
        '''
        m, n = obj_array.shape[0], det_array.shape[0]
        detxS = int(np.sqrt(n))
        detxL = int(detxS / scale)
        cnt = 0
        m_ptr = np.zeros((m + 1,))
        m_idx, n_idx, vec, attenuation = [], [], [], []
        for i in tqdm(range(m)):
            src = obj_array[i]
            l = self.calLength(src=src, ends=det_array, normals=self.ns, centers=self.cs)  # [k, n]
            attenuation_i = np.exp(-1 * np.sum(self.mus[:, None] * l, axis=0))  # [n,]
            # 按0.1mm 划分，现在进行聚合
            if not attenuation_i.flags['C_CONTIGUOUS']:
                attenuation_i = np.ascontiguousarray(attenuation_i)
            attenuation_i_group = (attenuation_i.reshape(detxL, scale, detxL, scale).sum(axis=(1, 3)) / (scale**2)).flatten()  # [2.5e3]
            attenuation.append(attenuation_i_group)
            # idx = np.where(attenuation_i < threshold)[0]
            # m_idx.extend([i]*idx.shape[0])
            # n_idx.extend(idx)
            # cnt += idx.shape[0]
            # m_ptr[i + 1] = cnt
            # vec_i = det_array - src[None, :]
            # vec.append(vec_i)
            # attenuation.append(attenuation_i)
        # m_ptr = np.array(m_ptr, dtype=np.int32)
        # m_idx = np.array(m_idx, dtype=np.int32)
        # n_idx = np.array(n_idx, dtype=np.int32)
        # vec = np.array(vec, dtype=np.float32).reshape((-1, 3))
        # attenuation = np.array(attenuation, dtype=np.float32).flatten()
        attenuation = np.array(attenuation, dtype=np.float64).flatten()
        if save:
            # np.save("./data/scatter_mptr.npy", m_ptr)
            # np.save("./data/scatter_nidx.npy", n_idx)
            np.save("./data/scatter_data.npy", attenuation)
            # np.save("./data/scatter_vec.npy", vec)
            # np.save("./data/scatter_midx.npy", m_idx)
        return attenuation
        # return {
        #     "m_ptr": m_ptr,
        #     "vec": vec,
        #     "data": attenuation,
        #     "m_idx": m_idx,
        #     "n_idx": n_idx,
        #     "shape": (m, n)
        # } 
                     
if __name__ == "__main__":
    tool = CalIntersectionLength(path="./Reconstruction/Vertex.yaml")
    srcs = np.array([
        [-5, 5, 10],
        [12, 5, 10]
    ])
    
    ends = np.array([
        [15, 5, 10],
        [-1, 5, 10]
    ]
    )
    
    re= tool.calLength(srcs[1], ends, normals=tool.ns, centers=tool.cs)
    print(re)
    # A = np.zeros((100, 100), dtype=int)

    # for i in range(10):       # 行方向子块索引
    #     for j in range(10):   # 列方向子块索引
    #         idx = i * 10 + j + 1  # 当前子块编号（1~100）
    #         A[i*10:(i+1)*10, j*10:(j+1)*10] = idx
    # A.flatten()