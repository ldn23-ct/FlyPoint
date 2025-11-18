import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.linear_model import RANSACRegressor, LinearRegression
# import tools.timeclass as timeclass
# import tools.parse_rail_motion as motion
import pandas as pd

class CalSysTool:
    '''
    Toolbox for computing system matrices
    '''
    def __init__(self,
                 src_pos: np.ndarray,  #shape: [3,]
                 grid_origin: np.ndarray,  #shape: [3,]
                 obj_size: np.ndarray,  #shape: [3,]
                 voxelsize: np.ndarray,  #shape: [3,]
                 slit: np.ndarray,  #shape:[2, 3]
                 det_center: np.ndarray,  #shape: [3,]
                 det_normal: np.ndarray,  #shape: [3,]
                 det_size: np.ndarray,  #shape: [2,]
                 pixel_sizeL: np.ndarray,  #shape: [2,]
                 fan_angle: np.float32,
                 angle_step,
                 E,
                 prob,
                 rho
                 ):
        self.src = src_pos
        self.obj_origin = grid_origin
        self.objsize = obj_size
        self.voxelsize = voxelsize
        self.slit = slit
        self.detsize = det_size
        self.pixelsizeL = pixel_sizeL
        self.detcenter = det_center
        self.detnormal = det_normal
        self.fan = fan_angle
        self.anglestep = angle_step
        self.E = E
        self.prob = prob / np.sum(prob)
        self.rho = rho
    
    def Fan2Euclidean(self, alpha:np.ndarray, d:np.ndarray) -> np.array:
        '''
        Conversion from Fanbeam coordinates to Euclidean coordinates
        
        Args:
            alpha(np.ndarray): ray incident angle(radians)
            d(np.ndarray): depth through the object

        Returns:
            pos(np.array): the spatial coordinates of the point at the current depth
        '''
        SOD = np.linalg.norm(self.obj_origin - self.src)
        z_vec = (self.obj_origin - self.src) / SOD
        y_vec = np.array([0, 1, 0])
        pos = d[:, None] * z_vec[None, :] + (np.tan(alpha) * (SOD + d))[:, None] * y_vec[None, :] + self.obj_origin
        return pos
    
    def CreateDetCoordinate(self, det_center=None, det_normal=None, det_size=None, v0=np.array([0, -1, 0])):
        '''
        Create a 2D detector coordinate system

        Args:
            det_center(np.ndarray): the spatial coordinates of the det center. shape of det_center: [3,]
            det_normal(np.ndarray): the normal vector of the det. shape of det_center: [3,]
            v1: one of the the standard basis. Assume that the detector is perpendicular to the horizontal plane.

        Returns:
            det_origin: the origin of the 2D detector coordinate system. shape: [3,]
            V: the standard basis for the 2D detector coordinate system. shape: [2, 3]
        '''
        if det_center == None: det_center = self.detcenter
        if det_normal == None: det_normal = self.detnormal
        if det_size ==None: det_size = self.detsize
        v1 = np.cross(det_normal, v0)
        det_corner0 = det_center - v0 * det_size[0] / 2 - v1 * det_size[1] / 2
        det_corner1 = det_center + v0 * det_size[0] / 2 - v1 * det_size[1] / 2
        det_corner2 = det_center + v0 * det_size[0] / 2 + v1 * det_size[1] / 2
        det_corner3 = det_center - v0 * det_size[0] / 2 + v1 * det_size[1] / 2
        self.detorigin = det_corner0
        self.detcorners = np.array([det_corner0, det_corner1, det_corner2, det_corner3])
        self.detv0, self.detv1 = v0, v1
        return det_corner0, np.array([v0, v1])

    def CreateDetArray(self, corners=None, pixelsize=None, detsize=None):
        '''
        Create a 2D array of detector points.

        Args:
            corners: the corners of the detector
            pixelsize: the size of the detector'pixel
            detsize: the size of the detector
        Returns:
            normal: the normals of the detector plane
            centers: the center points' position of the pixels
        '''
        if corners == None: corners = self.detcorners
        if pixelsize == None: pixelsize = self.pixelsizeL
        if detsize == None: detsize = self.detsize
        nx, ny = int(detsize[0] / pixelsize[0]), int(detsize[1] / pixelsize[1])
        P0, P1, P2, P3 = corners[0], corners[1], corners[2], corners[3]
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
        centers = (centers.transpose(1, 0, 2))
        return n/nn, centers

    def Euclidean2Det(self, pts):
        '''
        Conversion from Euclidean coordinates to Det coordinates

        Args:
            pts: points on the detector plane, expressed in 3D Euclidean coordinates
        
        Returns:
            pts: points on the detector plane, expressed in Det coordinates
        '''
        vec = pts - self.detorigin  # [n, 3]
        row = np.einsum('ij,j->i', vec, self.detv0)
        col = np.einsum('ij,j->i', vec, self.detv1)
        return np.stack((row, col), axis=1)

    def CalIntersectionLine(self, ds, slit=None, det_center=None, det_normal=None):
        '''
        Solving for intersection lines

        Args:
            ds(np.ndarray): depth list. shape of ds: [n,]
            slit(np.ndarray): the spatial coordinates of 2 points on the slit. shape of slit: [2, 3]
            det_center(np.ndarray): the spatial coordinates of the det center. shape of det_center: [3,]
            det_normal(np.ndarray): the normal vector of the det. shape of det_center: [3,]

        Returns:

        '''
        if slit == None: slit = self.slit
        if det_center == None: det_center = self.detcenter
        if det_normal == None: det_normal = self.detnormal

        pos = self.Fan2Euclidean(alpha=np.zeros_like(ds), d=ds)  # [n, 3]
        scatter_vec0 = slit[0] - pos  # [n, 3]
        scatter_vec1 = slit[1] - pos  # [n, 3]
        src2det_vec = det_center - pos  # [n, 3]
        t0 = np.einsum('ij,j->i', src2det_vec, det_normal) / np.einsum('ij,j->i', scatter_vec0, det_normal)
        t1 = np.einsum('ij,j->i', src2det_vec, det_normal) / np.einsum('ij,j->i', scatter_vec1, det_normal)
        p0 = pos + t0[:, None] * scatter_vec0
        p1 = pos + t1[:, None] * scatter_vec1
        p0_det = self.Euclidean2Det(p0)
        p1_det = self.Euclidean2Det(p1)
        cols = ((p0_det + p1_det) / 2)[:, 1]
        # print(cols[1:] - cols[:-1])
        cols = (cols / self.pixelsizeL[1]).astype(np.int64)
        max_col = int(self.detsize[1] / self.pixelsizeL[1])
        cols[cols >= max_col] = int(max_col - 1)
        return cols
    
    def klein_nishina(self, mu):
        """
        Klein-Nishina 微分散射截面
        
        Args:
            E   : 入射光子能量 keV  shape: [e,]
            prob: 不同能量光子的发射概率, 已归一化  shape: [e,]
            mu  : 散射角余弦 cos(theta)  shape: [n,]
            unit: "keV" 或 "MeV"(默认 "keV")
            
        Returns:
            dsdo: 微分截面 dσ/dΩ,单位 mm^2/sr   shape: [n,]
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
    
    def CalSystem(self):
        '''
        Calculate System Matrix

        Args:
            None

        Returns:
            sys: [p,d,m] 不同角度不同深度对探测器特定列的响应
            cols: [d,] 不同深度对应的探测器列数
        '''
        alphas = np.deg2rad(np.arange(-self.fan/2, self.fan/2 + self.anglestep, self.anglestep))[::-1]
        ds = np.arange(self.voxelsize[2]/2, self.objsize[2], self.voxelsize[2])
        P, D, M = alphas.shape[0], ds.shape[0], int(self.detsize[1] / self.pixelsizeL[1])
        self.CreateDetCoordinate()
        _, det_pts = self.CreateDetArray()
        cols = self.CalIntersectionLine(ds)
        det_pts_d = det_pts[:, cols, :]  # [M, D, 3] 第i列表示深度i下对应的那一列像素点空间坐标
        pixel_area = self.pixelsizeL[0] * self.pixelsizeL[1]
        sysmatrix = np.zeros((P, D, M))
        for p in range(P):
            alpha = alphas[p] * np.ones_like(ds)
            obj_pts = self.Fan2Euclidean(alpha, ds)  # [D, 3]
            
            #--------------散射角--------------#
            emit_vec = [0, np.sin(alphas[p]), -1*np.cos(alphas[p])]  #[3,] 射线朝负z方向发射
            scatter_vec = det_pts_d - obj_pts[None, :, :]
            r = np.linalg.norm(scatter_vec, axis=2)
            scatter_vec = scatter_vec / r[:, :, None]  #[M, D, 3]
            cos_scatter_angle = np.einsum('ijk,k->ij', scatter_vec, emit_vec)  #[M, D]
            kn = (self.klein_nishina(cos_scatter_angle.flatten(order='C'))).reshape(cos_scatter_angle.shape)
            kn_col_max = np.max(kn, axis=0)  # [D,]
            kn = kn / kn_col_max[None, :]
            #--------------散射角--------------#
            
            #--------------立体角--------------#
            cos_solid_angle = np.einsum('ijk,k->ij', scatter_vec, self.detnormal)
            cos_solid_angle = np.abs(np.clip(cos_solid_angle, -1.0, 1.0))  # [M, D]
            solid_angle = pixel_area * cos_solid_angle / np.square(r)
            solid_col_max = np.max(solid_angle, axis=0)
            solid_angle = solid_angle / solid_col_max[None, :]
            #--------------立体角--------------#
            
            coffei = kn * solid_angle
            col_sum = np.sum(coffei, axis=0)
            coffei = coffei / col_sum  # [M, D]
            sysmatrix[p, :, :] = np.transpose(coffei)
        return sysmatrix, cols
    
    def BackProjection(self, sys, cols, DetResponse):
        '''
        Args:
            sys: [p, d, m]  不同角度不同深度对特定列的响应
            cols: [d,]  不同深度对应的探测器列数
            detresponse: [p, m, n]  不同角度下的探测器响应
        Returns:
            back_value: [p, d]  不同角度不同深度的反投影值
        '''
        alphas = np.deg2rad(np.arange(-self.fan/2, self.fan/2 + self.anglestep, self.anglestep))[::-1]
        ds = np.arange(self.voxelsize[2]/2, self.objsize[2], self.voxelsize[2])
        P, D, M = alphas.shape[0], ds.shape[0], int(self.detsize[1] / self.pixelsizeL[1])
        back_value = np.zeros((P, D))
        # sys_ones = np.ones_like(sys)
        for i in range(D):
            if cols[i] < 0 or cols[i] > M - 1:
                back_value[:, i] = np.zeros(P)
            else:
                # temp = np.einsum('pm,pm->p', sys_ones[:, i, :], DetResponse[:, :, cols[i]])
                temp = np.einsum('pm,pm->p', sys[:, i, :], DetResponse[:, :, cols[i]])
                back_value[:, i] = temp
        return back_value

    def BackProjection1(self, prefunc, event):
        '''
        event: ['timestamps', 'x', 'y', 'rail_x', 'rail_y', 'angle']
        '''
        pos_z = prefunc(event[:, 2])
        start_d = self.src[2] - self.slit[0, 2] + 28  # 标定时钨板前表面到狭缝距离28mm
        pos = np.stack([event[:, 3], event[:, 4] + start_d*np.tan(np.deg2rad(event[:, 5])), pos_z], axis=1)
        
        # pos_z = prefunc(event[:, 1])
        # start_d = self.src[2] - self.slit[0, 2] + 28  # 标定时钨板前表面到狭缝距离28mm
        # pos = np.stack([event[:, 2], event[:, 3] + start_d*np.tan(np.deg2rad(event[:, 4])), pos_z], axis=1)
        # bbox = [-1, 1, -1, 1, 0, 60]
        
        # bbox = [-110, 10, 145, 241, 0, 60]
        bbox = [-100, 10, 200, 296, 0, 60]
        delta_x = 1
        delta_y = 2
        delta_z = 0.5
        Nx = int((bbox[1] - bbox[0]) / delta_x) + 1
        Ny = int((bbox[3] - bbox[2]) / delta_y) + 1
        Nz = int((bbox[5] - bbox[4]) / delta_z) + 1
        V = np.zeros((Nx, Ny, Nz))
        
        mask = (pos[:, 2] < 0) | (pos[:, 2] > 60)
        pos = pos[~mask]
        
        
        x_idx = np.round((pos[:, 0] - bbox[0])/delta_x).astype(int)
        y_idx = np.round((pos[:, 1] - bbox[2])/delta_y).astype(int)
        z_idx = np.round((pos[:, 2] - bbox[4])/delta_z).astype(int)
        np.add.at(V, (x_idx, y_idx, z_idx), 1)
        return V

    def trilerp_uniform(self, sys, x0, dx, y0, dy, z0, dz, pos, clamp=True, fill_value=np.nan):
        """
        sys: (Px, Py, Pz) 连续内存（C-order）建议
        pos: (N,3) 物理坐标，列顺序对应 (x,y,z)
        clamp=True  -> 越界点夹到边界（近邻外推）
        clamp=False -> 越界返回 fill_value
        """
        sys = np.asarray(sys)
        pos = np.asarray(pos, dtype=np.float64)
        Px, Py, Pz = sys.shape

        # 连续索引坐标（不取整）
        ix_f = (pos[:,0] - x0) / dx
        iy_f = (pos[:,1] - y0) / dy
        iz_f = (pos[:,2] - z0) / dz

        i0 = np.floor(ix_f).astype(np.int64)
        j0 = np.floor(iy_f).astype(np.int64)
        k0 = np.floor(iz_f).astype(np.int64)

        if clamp:
            i0 = np.clip(i0, 0, Px-2)
            j0 = np.clip(j0, 0, Py-2)
            k0 = np.clip(k0, 0, Pz-2)
            tx = np.clip(ix_f - i0, 0.0, 1.0)
            ty = np.clip(iy_f - j0, 0.0, 1.0)
            tz = np.clip(iz_f - k0, 0.0, 1.0)
            valid = np.ones_like(tx, dtype=bool)
        else:
            valid = (i0>=0)&(i0<Px-1)&(j0>=0)&(j0<Py-1)&(k0>=0)&(k0<Pz-1)
            # 为避免越界取值，复制夹取索引，但后面用 valid 回填
            i0c = np.clip(i0, 0, Px-2)
            j0c = np.clip(j0, 0, Py-2)
            k0c = np.clip(k0, 0, Pz-2)
            tx = ix_f - i0
            ty = iy_f - j0
            tz = iz_f - k0
            i0, j0, k0 = i0c, j0c, k0c
            tx = np.where(valid, tx, 0.0)
            ty = np.where(valid, ty, 0.0)
            tz = np.where(valid, tz, 0.0)

        i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1

        V000 = sys[i0, j0, k0]; V100 = sys[i1, j0, k0]
        V010 = sys[i0, j1, k0]; V110 = sys[i1, j1, k0]
        V001 = sys[i0, j0, k1]; V101 = sys[i1, j0, k1]
        V011 = sys[i0, j1, k1]; V111 = sys[i1, j1, k1]

        c00 = V000*(1-tx) + V100*tx
        c10 = V010*(1-tx) + V110*tx
        c01 = V001*(1-tx) + V101*tx
        c11 = V011*(1-tx) + V111*tx
        c0  = c00*(1-ty) + c10*ty
        c1  = c01*(1-ty) + c11*ty
        vals = c0*(1-tz) + c1*tz

        if not clamp:
            vals = np.where(valid, vals, fill_value)
        return vals

    def VoxelInterpolation(self,
                           points_xy, values,
                           delta_x, delta_y,
                           *,  # 强制使用命名参数
                           sigma_x_phys=None, sigma_y_phys=None,
                           sigma_x_pix=None,  sigma_y_pix=None,
                           cval=0.0,
                           eps=1e-8
                        ):
        if points_xy == None:
            alphas = np.deg2rad(np.arange(-self.fan/2, self.fan/2 + self.anglestep, self.anglestep))[::-1]
            ds = np.arange(self.voxelsize[2]/2, self.objsize[2], self.voxelsize[2])
            pos = np.zeros((alphas.shape[0], ds.shape[0], 3))
            for p in range(alphas.shape[0]):
                alpha = alphas[p] * np.ones_like(ds)
                obj_pts = self.Fan2Euclidean(alpha, ds)
                pos[p, :, :] = obj_pts
            points_xy = pos.reshape(-1, 3)[:, 1:]
        x = points_xy[:, 0]
        y = points_xy[:, 1]    
        
        SOD = np.abs(self.obj_origin[2] - self.src[2])
        slice_halfy = (SOD + self.objsize[2]) * np.tan(np.deg2rad(self.fan) / 2)
        ny = int(2 * np.ceil((slice_halfy - self.voxelsize[1]/2) / self.voxelsize[1])) + 1
        nz = int(np.ceil(self.objsize[2] / self.voxelsize[2]))
        obj_slice_size = [ny * self.voxelsize[1], self.objsize[2]]
        vec = -1  #  物体从负方向开始
        x_max = obj_slice_size[0] /2
        x_min = -1 * x_max
        y_max = self.obj_origin[2]
        y_min = y_max + vec * obj_slice_size[1]
        Nx = int(np.ceil((x_max - x_min) / delta_x))
        Ny = int(np.ceil((y_max - y_min) / delta_y))
        
        # --- 3) 连续坐标 → 像素索引（浮点，以像素中心为对齐） ---
        fx = (x - x_min) / delta_x - 0.5
        fy = (y_max - y) / delta_y - 0.5

        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        di = fx - i0
        dj = fy - j0

        # 4 邻双线性权重
        w00 = (1 - di) * (1 - dj)
        w10 = di * (1 - dj)
        w01 = (1 - di) * dj
        w11 = di * dj
        
        # # 4 只对竖直方向插值，深度方向不做插值
        # w0 = 1.0 - di
        # w1 = di

        S = np.zeros((Nx, Ny), dtype=np.float64)
        W = np.zeros((Nx, Ny), dtype=np.float64)

        def safe_add(arr, ii, jj, ww, val=None):
            mask = (ii >= 0) & (jj >= 0) & (ii < Nx) & (jj < Ny) & (ww > 0)
            if np.any(mask):
                if val is None:
                    np.add.at(arr, (ii[mask], jj[mask]), ww[mask])
                else:
                    np.add.at(arr, (ii[mask], jj[mask]), ww[mask] * val[mask])

        # def safe_add_rc(arr, ii, jj, ww, val=None):
        #     mask = (ii >= 0) & (jj >= 0) & (ii < Nx) & (jj < Ny) & (ww > 0)
        #     if np.any(mask):
        #         if val is None:
        #             np.add.at(arr, (ii[mask], jj[mask]), ww[mask])
        #         else:
        #             np.add.at(arr, (ii[mask], jj[mask]), ww[mask] * val[mask])

        # 双线性桶化：按 (行 i, 列 j) 累加
        safe_add(S, i0,   j0,   w00, values)
        safe_add(S, i0+1, j0,   w10, values)
        safe_add(S, i0,   j0+1, w01, values)
        safe_add(S, i0+1, j0+1, w11, values)

        safe_add(W, i0,   j0,   w00)
        safe_add(W, i0+1, j0,   w10)
        safe_add(W, i0,   j0+1, w01)
        safe_add(W, i0+1, j0+1, w11)
        
        # # 栅格化（仅行方向线性，两行；列为最近邻）
        # safe_add_rc(S, i0,   j0, w0, values)
        # safe_add_rc(S, i0+1, j0, w1, values)
        # safe_add_rc(W, i0,   j0, w0)
        # safe_add_rc(W, i0+1, j0, w1)
        
        # 计算各向异性 sigma（像素单位）
        if sigma_x_pix is None or sigma_y_pix is None:
            if (sigma_x_phys is None) or (sigma_y_phys is None):
                # 没给就默认 0.6 像素（保守）
                sigma_x_pix = 0.6 if sigma_x_pix is None else sigma_x_pix
                sigma_y_pix = 0.6 if sigma_y_pix is None else sigma_y_pix
            else:
                sigma_x_pix = sigma_x_phys / delta_x   # 行方向
                sigma_y_pix = sigma_y_phys / delta_y   # 列方向
                
        # 各向异性高斯卷积：顺序= (行sigma, 列sigma) = (sigma_x_pix, sigma_y_pix)
        S_blur = gaussian_filter(S, sigma=(sigma_x_pix, sigma_y_pix),
                                mode="reflect", cval=cval)
        W_blur = gaussian_filter(W, sigma=(sigma_x_pix, sigma_y_pix),
                                mode="reflect", cval=cval)
        # 仅在 axis=0（行轴）卷积；列轴不卷积（保持列分辨率）
        # S_blur = gaussian_filter1d(S, sigma=sigma_x_pix, axis=0,
        #                         mode="reflect", cval=cval)
        # W_blur = gaussian_filter1d(W, sigma=sigma_x_pix, axis=0,
        #                         mode="reflect", cval=cval)
        V = S_blur / (W_blur + eps)
        # V = S_blur
        return V

    def VoxelInterpolation3D(self,
        points_xyz, values,
        delta_x, delta_y, delta_z,
        *,
        # 三选一：给“物理单位”或“像素单位”的 sigma；缺省时默认 0.6 像素
        sigma_x_phys=None, sigma_y_phys=None, sigma_z_phys=None,
        sigma_x_pix=None,  sigma_y_pix=None,  sigma_z_pix=None,
        bbox=None,                  # (xmin,xmax, ymin,ymax, zmin,zmax) —— 必填
        boundary_mode="reflect",    # 'reflect'|'nearest'|'mirror'|'wrap'|'constant'
        cval=0.0,
        eps=1e-8
    ):
        """
        3D：三线性桶化 + 各向异性 3D 高斯卷积 + S/W 归一化（无外延，输出体素严格覆盖 bbox）
        轴顺序：V.shape == (Nx, Ny, Nz)  对应 x,y,z 三轴
        """
        assert bbox is not None and len(bbox) == 6, "bbox=(xmin,xmax,ymin,ymax,zmin,zmax) 必须提供"

        x = points_xyz[:, 0]  # 轴0
        y = points_xyz[:, 1]  # 轴1
        z = points_xyz[:, 2]  # 轴2
        v = values

        xmin, xmax, ymin, ymax, zmin, zmax = bbox

        # 体素数（严格覆盖 bbox）
        Nx = int(np.ceil((xmax - xmin) / delta_x))
        Ny = int(np.ceil((ymax - ymin) / delta_y))
        Nz = int(np.ceil((zmax - zmin) / delta_z))

        # 连续坐标 -> 以体素中心 (i+0.5, j+0.5, k+0.5) 对齐的浮点索引
        fi = (x - xmin) / delta_x - 0.5
        fj = (y - ymin) / delta_y - 0.5
        fk = (z - zmin) / delta_z - 0.5

        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        k0 = np.floor(fk).astype(int)
        di = fi - i0
        dj = fj - j0
        dk = fk - k0

        # 8 邻三线性权重
        w000 = (1-di)*(1-dj)*(1-dk)
        w100 = di    *(1-dj)*(1-dk)
        w010 = (1-di)*dj    *(1-dk)
        w110 = di    *dj    *(1-dk)
        w001 = (1-di)*(1-dj)*dk
        w101 = di    *(1-dj)*dk
        w011 = (1-di)*dj    *dk
        w111 = di    *dj    *dk

        S = np.zeros((Nx, Ny, Nz), dtype=np.float64)  # (x,y,z)
        W = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        def add8(arr, val=None):
            # 将 8 个角加进去，自动裁掉越界
            def safe_add(ii, jj, kk, ww, val_):
                mask = (ii>=0)&(jj>=0)&(kk>=0)&(ii<Nx)&(jj<Ny)&(kk<Nz)&(ww>0)
                if not np.any(mask): return
                if val_ is None:
                    np.add.at(arr, (ii[mask], jj[mask], kk[mask]), ww[mask])
                else:
                    np.add.at(arr, (ii[mask], jj[mask], kk[mask]), ww[mask]*val_[mask])
            safe_add(i0,   j0,   k0,   w000, val)
            safe_add(i0+1, j0,   k0,   w100, val)
            safe_add(i0,   j0+1, k0,   w010, val)
            safe_add(i0+1, j0+1, k0,   w110, val)
            safe_add(i0,   j0,   k0+1, w001, val)
            safe_add(i0+1, j0,   k0+1, w101, val)
            safe_add(i0,   j0+1, k0+1, w011, val)
            safe_add(i0+1, j0+1, k0+1, w111, val)

        add8(S, v)
        add8(W, None)

        # 计算各向异性 sigma（像素单位）
        use_pix = (sigma_x_pix is not None) and (sigma_y_pix is not None) and (sigma_z_pix is not None)
        use_phys = (sigma_x_phys is not None) and (sigma_y_phys is not None) and (sigma_z_phys is not None)

        if not (use_pix or use_phys):
            # 全缺省：保守默认 0.6 像素
            sigma_x_pix = 0.6; sigma_y_pix = 0.6; sigma_z_pix = 0.6
        elif use_phys and not use_pix:
            sigma_x_pix = sigma_x_phys / delta_x
            sigma_y_pix = sigma_y_phys / delta_y
            sigma_z_pix = sigma_z_phys / delta_z
        # 若已给 pix，就直接用

        # 3D 各向异性高斯卷积（顺序与数组轴一致：(x,y,z) -> (sigma_x, sigma_y, sigma_z)）
        S_blur = gaussian_filter(S, sigma=(sigma_x_pix, sigma_y_pix, sigma_z_pix),
                                mode=boundary_mode, cval=cval)
        W_blur = gaussian_filter(W, sigma=(sigma_x_pix, sigma_y_pix, sigma_z_pix),
                                mode=boundary_mode, cval=cval)

        V = S_blur / (W_blur + eps)

        # 返回地理信息（左下前原点 + 体素尺寸）
        return V, (xmin, ymin, zmin, delta_x, delta_y, delta_z)

    def Pos2DetRes(self, filepath, W=512, H=512):
        pos = np.load(filepath) [:, 1:3]
        pos = pos[~np.isnan(pos).any(axis=1)]
        yi = pos[:,0].astype(np.int64, copy=False)
        xi = pos[:,1].astype(np.int64, copy=False)
        m = (xi>=0) & (xi<W) & (yi>=0) & (yi<H)
        xi = xi[m]; yi = yi[m]
        lin = yi * W + xi
        img = np.bincount(lin, minlength=W*H).reshape(H, W)
        return img[:, ::-1]

    def PreCols(self, ds, cols, ransac=True, max_trials=200, residual_threshold=None, eps=1e-12):
        '''
        预测不同列对应的深度，依赖标定数据
        
        Args:
            d: 标定模体交界面对应深度序列
            cols: 探测器响应交界面对应列数
        Returns:
            pre_col: 用于预测深度对应列数的函数
        '''
        ds = np.asarray(ds, dtype=float).ravel()
        cols = np.asarray(cols, dtype=float).ravel()
        delta_cols = cols[1:] - cols[0]
        x = 1.0 / delta_cols
        y = 1.0 / ds[1:]
    
        A = B = None
        used_ransac = False
        info = {}
        if ransac:
            # 自适应阈值：y 的IQR比例，防止尺度问题
            if residual_threshold is None:
                q1, q3 = np.percentile(y, [25, 75])
                iqr = max(q3 - q1, eps)
                residual_threshold = 1.5 * iqr

            base = LinearRegression(fit_intercept=True)
            ransac_model = RANSACRegressor(
                estimator=base,
                max_trials=max_trials,
                residual_threshold=residual_threshold,
                random_state=0
            )
            ransac_model.fit(x.reshape(-1, 1), y)
            B = float(ransac_model.estimator_.coef_[0])     # slope
            A = float(ransac_model.estimator_.intercept_)   # intercept
            used_ransac = True
            inlier_mask = ransac_model.inlier_mask_
            info["inliers"] = int(np.sum(inlier_mask))
            info["outliers"] = int(len(inlier_mask) - info["inliers"])

        if A is None or B is None:
            # 退化为普通最小二乘
            # y = A + B x
            B, A = np.polyfit(x, y, 1)
            used_ransac = False      
        
        def pre_col(col_new, eps=1e-18):
            delta_col = col_new - cols[0]
            denom = A * delta_col + B
            return delta_col / denom
        
        model = (A, B)
        
        return pre_col, model


def bins_count_image_from_xy(pos, W=512, H=512, round_mode="truncate"):
    """
    对一组[y,x]坐标做分箱计数，返回HxW整型图。
    round_mode:
      - 'truncate': 直接astype(int)，向0截断（与你原代码一致）
      - 'round':    np.rint 四舍五入
      - 'floor':    np.floor 向下取整
    """
    if pos.size == 0:
        return np.zeros((H, W), dtype=np.int64)

    x = pos[:, 0]
    y = pos[:, 1]

    if round_mode == "truncate":
        yi = y.astype(np.int64, copy=False)
        xi = x.astype(np.int64, copy=False)
    elif round_mode == "round":
        yi = np.rint(y).astype(np.int64, copy=False)
        xi = np.rint(x).astype(np.int64, copy=False)
    elif round_mode == "floor":
        yi = np.floor(y).astype(np.int64, copy=False)
        xi = np.floor(x).astype(np.int64, copy=False)
    else:
        raise ValueError("round_mode ∈ {'truncate','round','floor'}")

    # 视野内筛选
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if not np.any(m):
        return np.zeros((H, W), dtype=np.int64)

    xi = xi[m]
    yi = yi[m]

    # 线性索引 + bincount（高效且无锁竞争）
    lin = yi * W + xi
    img = np.bincount(lin, minlength=W * H).reshape(H, W)

    # 与你原函数保持一致：左右翻转
    return img


def func0(A, B, col0):
    def pre_col(col_new, eps=1e-18):
        delta_col = col_new - col0
        denom = A * delta_col + B
        return delta_col / denom
    return pre_col, (A, B)


if __name__ == "__main__":
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    slit1 = np.array([[28, -25, -37], [28, 25, -37]])  # 靠近射线源，对应探测器后半段
    slit2 = np.array([[48.12, -25, -37], [48.12, 25, -37]])  # 远离射线源，对应探测器前半段
    src = np.array([0, 0, 158])
    obj_origin = np.array([0, 0, -50])
    objsize = np.array([200, 200, 70])
    fan = 15
    angle_step = 1
    voxelsize = np.array([5, 5, 0.1])
    det_size = np.array([50, 50])
    # pixelsizeL = np.array([1, 1])
    pixelsizeL = np.array([50/512, 50/512])
    det_center = np.array([-77.04, 0, 37.58])
    det_normal = np.array([0.7661, 0, -0.6427])
    det_center1 = np.array([66.42, 0, -5.44])
    det_normal1 = np.array([-0.4291, 0, -0.9033])
    E = np.array([160])
    prob = np.array([1])
    voxelshape = (objsize / voxelsize).astype(np.int32)
    rho = np.zeros(voxelshape)
    rho.fill(0.134*2.7*0.1)

    toolbox = CalSysTool(src_pos=src,
                         grid_origin=obj_origin,
                         obj_size=objsize,
                         voxelsize=voxelsize,
                        #  slit=slit,
                        #  slit=slit1,
                         slit=slit2,
                        #  det_center=det_center,
                         det_center=det_center1,
                        #  det_normal=det_normal,
                         det_normal=det_normal1,
                         det_size=det_size,
                         pixel_sizeL=pixelsizeL,
                         fan_angle=fan,
                         angle_step=angle_step,
                         E=E,
                         prob=prob,
                         rho=rho)
    

    cols_calibration = np.load("cols.npy")[::-1]
    ds_calibration = np.arange(0, 62, 2)
    prefunc, model = toolbox.PreCols(ds_calibration, cols_calibration)
    # prefunc, model = func0(0.0003, 4.8062, 90)  #slit2
    # prefunc = func0(-0.0672, 8.4804, 358)  #slit1
    
    A, B = model
    ds = np.arange(0, 60)
    dcs = B*ds / (1-A*ds)
    cs = cols_calibration[0] + dcs

    # npy_path = './TrueData/test/2025_11_13_15_15_11/2_events_with_angle.npy'
    # event = np.load(npy_path)
    # event[:, 1] = 511 - event[:, 1]  # 双缝位置需要镜像
    # df = pd.read_csv(csv_path, usecols=["x", "y", 'rail_x', 'rail_y', 'angle'])
    # df = df.dropna(subset=["x", "y", 'rail_x', 'rail_y', 'angle'])
    # event = df.to_numpy()[:, [0, 1]]
    # img = bins_count_image_from_xy(event, round_mode="round")
    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(np.arange(img.shape[1]), np.sum(img, axis=1))
    # plt.imshow(img, cmap='gray')
    # for i in range(cs.shape[0]):
        # axes[0].axvline(x=cs[i])
    
    
    
    
    # col = np.arange(69, 500)
    # d_pre = prefunc(col)
    # d_pre[np.isinf(d_pre)] = 0

    # sys, cols = toolbox.CalSystem()

    P = int(fan / angle_step) + 1
    M = int(det_size[0] / pixelsizeL[0])
    # DetResponse = np.zeros((P, M, M))
    # deg_0_detresponse = toolbox.Pos2DetRes("./TrueData/daq/2025_10_29_15_26_18/position.npy")
    # deg_6_detresponse = toolbox.Pos2DetRes("./TrueData/daq/2025_10_29_15_33_37/position.npy")
    # deg_neg6_detresponse = toolbox.Pos2DetRes("./TrueData/daq/2025_10_29_15_40_4/position.npy")

    # fig, axes = plt.subplots(3, 2, figsize=(12,18), constrained_layout=True)
    # axes[0, 0].imshow(deg_0_detresponse, cmap='gray', aspect='equal')
    # axes[1, 0].imshow(deg_6_detresponse, cmap='gray', aspect='equal')
    # axes[2, 0].imshow(deg_neg6_detresponse, cmap='gray', aspect='equal')
    # axes[0, 0].scatter(np.arange(M), np.sum(deg_0_detresponse, axis=0))
    # axes[1, 0].scatter(np.arange(M), np.sum(deg_6_detresponse, axis=0))
    # axes[2, 0].scatter(np.arange(M), np.sum(deg_neg6_detresponse, axis=0))

    # angle_response = [deg_0_detresponse, deg_6_detresponse, deg_neg6_detresponse]
    # angle_idx = [7, 1, 13]
    # for i in range(3):
    #     DetResponse = np.zeros((P, M, M))
    #     DetResponse[angle_idx[i], :, :] = angle_response[i]
    #     # values = toolbox.BackProjection(sys, cols, DetResponse)
    #     values = toolbox.BackProjection(sys, cols_pre.astype(np.int32), DetResponse)
    #     values = values.flatten()
    #     V = toolbox.VoxelInterpolation(None, values,
    #                             delta_x=toolbox.voxelsize[1], delta_y=toolbox.voxelsize[2],
    #                             sigma_x_phys=0.6 * toolbox.voxelsize[1],
    #                             sigma_y_phys=0.4 * toolbox.voxelsize[2])
    #     axes[i, 1].imshow(V, cmap="gray", aspect='auto')
    #     axes[i, 1].scatter(np.arange(V.shape[1]), np.sum(V, axis=0))
        
    # csv_path = './TrueData/tasks/2025-10-29_16-09_5f844ddc/data/events_with_angle.csv'
    # xy_groups = timeclass.load_xy_groups(csv_path, 0, 10)
    # for i in range(16):
    #     img = timeclass.bins_count_image_from_yx(xy_groups[i])
    #     DetResponse[i, :, :] = img
    # values = toolbox.BackProjection(sys, cols_pre.astype(np.int32), DetResponse)    
    # values = values.flatten()    
    # V = toolbox.VoxelInterpolation(None, values,
    #                         delta_x=toolbox.voxelsize[1], delta_y=toolbox.voxelsize[2],
    #                         sigma_x_phys=0.6 * toolbox.voxelsize[1],
    #                         sigma_y_phys=0.4 * toolbox.voxelsize[2])
    # vmin, vmax = np.percentile(V, [10, 90])
    # plt.imshow(V, cmap="gray", aspect='auto', vmin=vmin, vmax=vmax) 
    # plt.show()
    
    
    event = np.load("./TrueData/task/2025-11-12_19-01_11e9aa61/events_with_angle.npy")
    # event[:, 2] = 511 - event[:, 2]  # 双缝位置需要镜像
    
    V = toolbox.BackProjection1(prefunc, event)
    # print(V.shape)
    # axes[1].plot(np.arange(V.shape[0]), V)
    # plt.show()
    # V, model = toolbox.BackProjection1(prefunc, event)
    # print(V.shape)
    # plt.imshow(np.sum(V, axis=1), cmap='gray', aspect='auto')
    # plt.show()
    for i in range(V.shape[2]):
        img = V[:, :, i]
        # axes[1].plot(np.arange(img.shape[1]), np.sum(img, axis=0))
        plt.imshow(img, cmap='gray', aspect='auto')
        # plt.show()
        plt.savefig(f"./fig/{i}.tif")
    
        
    