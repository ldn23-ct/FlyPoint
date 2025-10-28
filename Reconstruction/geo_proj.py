import numpy as np
from tqdm import tqdm

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
    centers = (centers.transpose(1, 0, 2))
    return n/nn, centers

def proj(
    det_array: np.ndarray,   # [n, n, 3]
    slit: np.ndarray,        # [2, 3] -> two points (S0, S1) defining an infinite line
    src: np.ndarray,         # [3,]
    alpha: float,            # radians; ray lies in yz-plane; angle w.r.t. +z
    eps_n: float = 1e-12,    # threshold for |n| (plane degeneracy: point ~ on line)
    eps_den: float = 1e-14,  # threshold for |n·v| (ray || plane)
    eps_num: float = 1e-14,  # threshold for |n·(src-S0)| (ray in plane test)        
):
    ni, nj, _ = det_array.shape
    s0, s1 = slit[0], slit[1]
    w = s1 - s0
    v = np.array([0.0, np.sin(alpha), np.cos(alpha)], dtype=float)

    p_s0 = det_array - s0[None, None, :]
    nvec = np.cross(w[None, None, :], p_s0)  
    n_norm = np.linalg.norm(nvec, axis=-1)
    nvec = nvec / n_norm[:, :, None]

    src_s0 = src - s0
    numerator = np.einsum('ijk,k->ij', nvec, src_s0)
    denominator = np.einsum('ijk,k->ij', nvec, v)

    # Initialize outputs
    t = np.full((ni, nj), np.nan, dtype=float)
    X = np.full((ni, nj, 3), np.nan, dtype=float)
    code = np.zeros((ni, nj), dtype=int)               # 0 = OK; others set below
    mask = np.zeros((ni, nj), dtype=bool)

    # Degenerate planes: |n| ~ 0 -> P lies on the slit line (no unique plane)
    deg_plane = (n_norm <= eps_n)
    code[deg_plane] = 1

    # Ray parallel to plane: |den| ~ 0
    par = (np.abs(denominator) <= eps_den) & (~deg_plane)
    # Among parallels, check if ray lies in plane: |num| ~ 0
    in_plane = par & (np.abs(numerator) <= eps_num)
    only_parallel = par & (~in_plane)

    code[only_parallel] = 2
    code[in_plane] = 3

    # Regular cases: unique intersection
    ok = (~deg_plane) & (~par)

    t[ok] = - numerator[ok] / denominator[ok]
    X[ok] = src[None, None, :] + t[ok][..., None] * v[None, None, :]

    mask[ok] = True

    return X, t, mask, code

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
        self.fan = np.deg2rad(fan_angle)
    
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
    
    def CreateDetCoordinate(self, det_center=None, det_normal=None, det_size=None, v0=np.array([0, 0, -1])):
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
        if det_center == None: slit = self.detcenter
        if det_normal == None: slit = self.detnormal
        if det_size ==None: det_size = self.detsize
        v1 = np.cross(det_normal, v0)
        det_corner0 = det_center - v0 * det_size[0] - v1 * det_size[1]
        det_corner1 = det_center + v0 * det_size[0] - v1 * det_size[1]
        det_corner2 = det_center + v0 * det_size[0] + v1 * det_size[1]
        det_corner3 = det_center - v0 * det_size[0] + v1 * det_size[1]
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
        row = vec * self.detv0[None, :]
        col = vec * self.detv1[None, :]
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
        if det_center == None: slit = self.detcenter
        if det_normal == None: slit = self.detnormal

        pos = self.Fan2Euclidean(alpha=np.zeros_like(ds), d=ds)  # [n, 3]
        scatter_vec0 = slit[0] - pos  # [n, 3]
        scatter_vec1 = slit[1] - pos  # [n, 3]
        src2det_vec = det_center - pos  # [n, 3]
        t0 = (src2det_vec * det_normal[None, :]) / (scatter_vec0 * det_normal[None, :])
        t1 = (src2det_vec * det_normal[None, :]) / (scatter_vec1 * det_normal[None, :])
        p0 = pos + t0 * scatter_vec0
        p1 = pos + t1 * scatter_vec1
        p0_det = self.Euclidean2Det(p0)
        p1_det = self.Euclidean2Det(p1)
        cols = ((p0_det + p1_det) / 2)[1]
        cols = int(cols / self.pixelsizeL[1])
        max_col = int(self.detsize[1] / self.pixelsizeL[1])
        cols[cols >= max_col] = int(max_col - 1)
        return cols
    
    def CalSystem(self):





if __name__ == "__main__":
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    src = np.array([0, 0, 158])
    obj_origin = np.array([0, 0, -50])
    objsize = np.array([200, 200, 70])
    fan = 15
    voxelsize = np.array([5, 5, 0.1])
    det_size = np.array([50, 50])
    pixelsizeS = np.array([0.1, 0.1])
    pixelsizeL = np.array([1, 1])

    toolbox = CalSysTool(src_pos=src,
                         grid_origin=obj_origin,
                         obj_size=objsize,
                         voxelsize=voxelsize,
                         slit=slit,
                         det_center=None,
                         det_normal=None,
                         det_size=det_size,
                         pixel_sizeL=pixelsizeL,
                         pixel_sizeS=pixelsizeS,
                         fan_angle=fan)
    
    ds = np.arange(0, 1, 1)
    pos = toolbox.Fan2Euclidean(alpha=np.zeros_like(ds)+np.deg2rad(30), d=ds)
    print(pos)

