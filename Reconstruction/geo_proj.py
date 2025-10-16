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

if __name__ == "__main__":
    # corners = [[20, 10, 20], [20, -10, 20], [20, -10, 0], [20, 10, 0]]
    # _, det = DetArray(corners, [2, 2], [20, 20])

    # slit = np.array([[10.0, -10.0, 10.0],
    #                  [10.0, +10.0, 10.0]])
    # src = np.array([0.0, 0.0, -50.0])
    # alpha = np.deg2rad(20.0)  # 与 +z 夹角 20°

    det_size = np.array([50, 50])
    pixelsizeS = np.array([0.1, 0.1])
    pixelsizeL = np.array([1, 1])
    det_corners = [[-58.19, 25, 52.51], [-58.19, -25, 52.51], [-90.33, -25, 14.2], [-90.33, 25, 14.2]]
    _, det = DetArray(det_corners, pixelsizeS, det_size)
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    # slit = np.array([-28.3, -37.35], [-27.65, -36.59])
    src = np.array([0, 0, 158])
    alpha = 0
    X, t, mask, code = proj(det, slit, src, alpha)
    print(X[0, 0:10, :])
    print(X[0, -10:-1, :])
    print(X[-1, 0:10, :])
    print(X[-1, -10:-1, :])
