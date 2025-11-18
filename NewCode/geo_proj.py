import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor, LinearRegression

class CalSysTool:
    '''
    Toolbox for computing system matrices
    '''
    def __init__(self,
                 src_pos: np.ndarray,  # 放射源坐标
                 slit: np.ndarray,
                 calibrationdata_path: str,  # 深度校正文件
                 ):
        self.src = src_pos
        self.slit = slit
        
    def BackProjection(self,
                       prefunc,
                       event,
                       bbox,
                       delta_x, delta_y, delta_z):
        '''
        event: ['timestamps', 'x', 'y', 'rail_x', 'rail_y', 'angle']
        '''
        pos_z = prefunc(event[:, 2])
        start_d = self.src[2] - self.slit[0, 2] + 28  # 标定时钨板前表面到狭缝距离28mm
        pos = np.stack([event[:, 3], event[:, 4] + (start_d + pos_z)*np.tan(np.deg2rad(event[:, 5])), pos_z], axis=1)

        Nx = int((bbox[1] - bbox[0]) / delta_x) + 1
        Ny = int((bbox[3] - bbox[2]) / delta_y) + 1
        Nz = int((bbox[5] - bbox[4]) / delta_z) + 1
        V = np.zeros((Nx, Ny, Nz))
        
        mask = (pos[:, 2] < 0) | (pos[:, 2] > 60)
        pos = pos[~mask]
        print(np.max(pos[:, 0]))
        print(np.min(pos[:, 0]))
        print(np.max(pos[:, 1]))
        print(np.min(pos[:, 1]))
          
        x_idx = np.round((pos[:, 0] - bbox[0])/delta_x).astype(int)
        y_idx = np.round((pos[:, 1] - bbox[2])/delta_y).astype(int)
        z_idx = np.round((pos[:, 2] - bbox[4])/delta_z).astype(int)
        np.add.at(V, (x_idx, y_idx, z_idx), 1)
        return V

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

def find_peaks_1d(
    x,
    window_size=5,
    threshold=None,
    return_smooth=False
):
    """
    一维数据寻峰（先平滑再找峰）

    参数
    ----
    x : 1D array-like
        原始数据
    window_size : int, default=5
        平滑窗长度（滑动平均），必须为 >= 1 的整数。
        一般越大越能抹掉毛刺，但也会抹平窄峰。
    threshold : float or None, default=None
        峰值最小高度阈值（在平滑后数据上判断）。
        - 如果为 None，则自动设置为 mean + 0.5*std
    return_smooth : bool, default=False
        是否返回平滑后的数据

    返回
    ----
    peaks_idx : np.ndarray
        峰位置的索引数组（在原始 x 的索引）
    smooth_x : np.ndarray (仅当 return_smooth=True 时返回)
        平滑后的数据
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        # 少于 3 个点不可能有局部极大值
        return (np.array([], dtype=int), x) if return_smooth else np.array([], dtype=int)

    # ---------- 1. 平滑去毛刺 ----------
    w = int(window_size)
    if w < 1:
        raise ValueError("window_size 必须 >= 1")
    if w > n:
        w = n  # 窗太大时，退而求其次用全局平均

    # 简单滑动平均
    kernel = np.ones(w, dtype=float) / w
    # 使用 'same' 保持长度不变；边缘自动做零填充
    smooth = np.convolve(x, kernel, mode="same")

    # ---------- 2. 自动阈值（如果没给） ----------
    if threshold is None:
        mu = smooth.mean()
        sigma = smooth.std()
        threshold = mu + 0.5 * sigma  # 经验阈值，你可以根据数据再调

    # ---------- 3. 寻找局部极大值 ----------
    # 条件： smooth[i] 比左右都大，并且超过阈值
    # 注意 i 从 1 到 n-2（因为要看 i-1 和 i+1）
    left = smooth[:-2]
    center = smooth[1:-1]
    right = smooth[2:]

    is_peak_center = (center > left) & (center >= right) & (center >= threshold)

    # 把中心索引转回全局索引：中心是 1..n-2
    peaks_idx = np.where(is_peak_center)[0] + 1

    if return_smooth:
        return peaks_idx, smooth
    else:
        return peaks_idx


if __name__ == "__main__":
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    slit1 = np.array([[28, -25, -37], [28, 25, -37]])  # 靠近射线源，对应探测器后半段
    slit2 = np.array([[48.12, -25, -37], [48.12, 25, -37]])  # 远离射线源，对应探测器前半段
    src = np.array([0, 0, 158])

    toolbox = CalSysTool(src_pos=src,
                         slit=slit,
                         calibrationdata_path=None,
                         )

    # 单缝标定模型
    cols_calibration = np.load("./data/Calibration_data/cols_calibration0.npy")
    ds_calibration = np.arange(0, 62, 2)
    prefunc0, model0 = toolbox.PreCols(ds_calibration, cols_calibration)
    
    # 双缝标定模型
    cols_calibration2 = np.load("./data/Calibration_data/cols_calibration1.npy")[:, 0]  #slit2
    cols_calibration1 = np.load("./data/Calibration_data/cols_calibration1.npy")[:, 1]  #slit1
    ds_calibration = np.arange(0, 32, 2)
    prefunc2, model2 = toolbox.PreCols(ds_calibration, cols_calibration2[:int(ds_calibration.shape[0])])
    prefunc1, model1 = toolbox.PreCols(ds_calibration, cols_calibration1[:int(ds_calibration.shape[0])])

    npy_path0 = './TrueData/test/2025-11-17_15-33_50eda15f/0_events_with_angle.npy'
    npy_path1 = './TrueData/test/2025-11-17_15-33_50eda15f/1_events_with_angle.npy'
    event0 = np.load(npy_path0) # ['timestamps', 'x', 'y', 'rail_x', 'rail_y', 'angle']
    event1 = np.load(npy_path1)
    
    # 根据目前标定单缝文件，重建位置距离狭缝 28mm--88mm
    bbox = [-82, 29, 219, 297, 0, 60]   # 间隔模体数据
    # bbox = [-104, -2, 138, 228, 0, 60]
    # bbox = [-140, -6, 218, 294, 0, 60]
    # bbox = [-5, 5, -5, 5, 0, 60]
    delta_x, delta_y, delta_z = 1, 1, 2
    V0 = toolbox.BackProjection(prefunc0,
                               event0,
                               bbox=bbox,
                               delta_x=delta_x,
                               delta_y=delta_y,
                               delta_z=delta_z
                               )
    V2 = toolbox.BackProjection(prefunc2,
                               event1,
                               bbox=bbox,
                               delta_x=delta_x,
                               delta_y=delta_y,
                               delta_z=delta_z
                               )
    V1 = toolbox.BackProjection(prefunc1,
                               event1,
                               bbox=bbox,
                               delta_x=delta_x,
                               delta_y=delta_y,
                               delta_z=delta_z
                               )
    Vs = [V0, V2, V1]
    titles = ["V0", "V2", "V1"]
    for i in range(V0.shape[2]):
        fig, axes = plt.subplots(1, len(Vs), figsize=(12, 4), constrained_layout=True)
        for j, ax in enumerate(axes):
            ax.imshow(Vs[j][:, :, i], cmap='gray', aspect='auto')
            ax.set_title(f"{titles[j]}  |  Slice {i*2}")
            ax.axis('off')
        plt.show()
    
        
    #region : 观测0--60mm深，每1mm对应的列数
    # A, B = model
    # img = bins_count_image_from_xy(event[:, [1, 2]], round_mode="round")
    
    # ds = np.arange(0, 60)
    # ds = np.array([35, 45, 55])
    # didx = ds / delta_z
    # dcs = B*ds / (1-A*ds)
    # cs = cols_calibration[0] + dcs
    
    # col_sum = np.sum(img, axis=1)
    # cs = find_peaks_1d(col_sum)
    # ds = prefunc(cs)
    # print(ds)
    # didx = ds / delta_z
    
    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(np.arange(img.shape[1]), np.sum(img, axis=1))
    # axes[0].imshow(img, cmap='gray', aspect='auto', interpolation='none')
    # axes[1].imshow(np.sum(V, axis=0), cmap='gray', aspect='auto', interpolation='none')
    # axes[1].plot(np.arange(121), np.sum(V, axis=(0, 1)))
    # for i in range(cs.shape[0]):
    #     axes[0].axvline(x=cs[i], color='red', linewidth=0.3, dashes=(5,2))
    #     axes[1].axvline(didx[i], color='red', linewidth=0.3, dashes=(5,2))
    # plt.show()
    #endregion
        




