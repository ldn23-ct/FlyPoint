import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                 pixel_sizeS: np.ndarray,  #shape:[2,]
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
        self.pixelsizeS = pixel_sizeS
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
        Create a 2D array of detector points

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
        alphas = np.deg2rad(np.arange(-self.fan/2, self.fan/2 + self.anglestep, self.anglestep))[::-1]
        ds = np.arange(self.voxelsize[2]/2, self.objsize[2], self.voxelsize[2])
        P, D = alphas.shape[0], ds.shape[0]
        back_value = np.zeros((P, D))
        for i in range(D):
            # det_response = DetResponse[cols[i]]
            temp = np.einsum('pm,m->p', sys[:, i, :], DetResponse[:, cols[i]])
            back_value[:, i] = temp
        return back_value

    def VoxelInterpolation(self,
        points_xy, values,
        delta_x, delta_y,
        *,  # 强制使用命名参数
        sigma_x_phys=None, sigma_y_phys=None,
        sigma_x_pix=None,  sigma_y_pix=None,
        bbox=None,
        pad_mult=2.5,
        eps=1e-8,
    ):
        SOD = np.abs(self.obj_origin[2] - self.src[2])
        slice_halfy = (SOD + self.objsize[2]) * np.tan(self.fan / 2)
        ny = int(2 * np.ceil((slice_halfy - self.voxelsize[1]/2) / self.voxelsize[1])) + 1
        nz = int(np.ceil(self.objsize[2] / self.voxelsize[2]))
        obj_slice_size = [ny * self.voxelsize[1], self.objsize[2]]
        vec = -1  #  物体从负方向开始
        y_start = (obj_slice_size[0] - self.voxelsize[1]) / 2
        z_start = self.obj_origin[2] + vec * self.voxelsize[2] / 2
        y_centers = [(y_start - i * self.voxelsize[1]) for i in range(ny)]
        z_centers = [(z_start + i * vec * self.voxelsize[2]) for i in range(nz)]
        Y, Z = np.meshgrid(y_centers, z_centers, indexing='ij')
        X = np.zeros_like(Y)
        self.obj_slice = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # 按行排序, 左上角为起点
        
        

if __name__ == "__main__":
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    src = np.array([0, 0, 158])
    obj_origin = np.array([0, 0, -50])
    objsize = np.array([200, 200, 70])
    fan = 15
    angle_step = 0.1
    voxelsize = np.array([5, 5, 0.1])
    det_size = np.array([50, 50])
    pixelsizeS = np.array([0.1, 0.1])
    pixelsizeL = np.array([1, 1])
    det_center = np.array([-74.26, 0, 33.355])
    det_normal = np.array([0.7661, 0, -0.6427])
    E = np.array([160])
    prob = np.array([1])
    voxelshape = (objsize / voxelsize).astype(np.int32)
    rho = np.zeros(voxelshape)
    rho.fill(0.134*2.7*0.1)

    toolbox = CalSysTool(src_pos=src,
                         grid_origin=obj_origin,
                         obj_size=objsize,
                         voxelsize=voxelsize,
                         slit=slit,
                         det_center=det_center,
                         det_normal=det_normal,
                         det_size=det_size,
                         pixel_sizeL=pixelsizeL,
                         pixel_sizeS=pixelsizeS,
                         fan_angle=fan,
                         angle_step=angle_step,
                         E=E,
                         prob=prob,
                         rho=rho)

    detResponses = np.load("./data/MC_data/0_degree_interval_5mm.npy")[:, ::-1]
    detResponses_nodefect = np.load("./data/MC_data//0_degree_no_defect.npy")[:, ::-1]
    res = detResponses / detResponses_nodefect - 1
    
    sys, cols = toolbox.CalSystem()
    back_value0 = toolbox.BackProjection(sys, cols, detResponses)
    back_value1 = toolbox.BackProjection(sys, cols, detResponses_nodefect)
    back_value2 = toolbox.BackProjection(sys, cols, res)
    x = np.arange(700)
    # plt.plot(x, back_value0[0])
    plt.plot(x, back_value0[75])
    # plt.plot(x, back_value1[75])
    # plt.plot(x, back_value2[75])
    plt.show()