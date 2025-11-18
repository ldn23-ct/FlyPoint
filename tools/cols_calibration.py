import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

def findcols(
    y,
    k,
    baseline_q=0.10,      # 用分位数估计底部
    smooth_window=5       # 移动平均窗口
):
    """
    在单通道 1D 曲线中，寻找 k 个峰的“左侧半高点”。

    假设：
        - 曲线中只有 k 个主要峰（可以有噪声毛刺）
        - 峰之间有一定间隔，不严重重叠

    Args:
        y: 1D array-like，离散概率或计数（不要求归一化）
        k: 期望的峰个数（整数 > 0）
        baseline_q: 用于估计底部的分位数（0~1），例如 0.1 表示 10% 分位
        smooth_window: 平滑窗口大小；<=1 表示不平滑

    Returns:
        x_halfs: shape (k,) 的浮点数组，每个元素是对应峰左侧半高点的 x（索引）
                 若某个峰无法找到半高点，对应元素为 -1.
        info: dict，包含一些中间结果，用于可视化/调试：
              - "y_smooth": 平滑后的曲线
              - "y_base":   基线高度（分位数）
              - "peaks_idx": 选中的 k 个峰的索引（按 x 从小到大）
              - "peaks_val": 对应峰值
              - "y_half":    对应每个峰的半高值数组
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    if n == 0:
        raise ValueError("Input y is empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # 1) 可选平滑（移动平均）
    if smooth_window is not None and smooth_window > 1:
        ker = np.ones(smooth_window, dtype=float) / smooth_window
        y_smooth = np.convolve(y, ker, mode="same")
    else:
        y_smooth = y

    # 2) 基线估计（分位数）
    y_base = np.quantile(y_smooth, baseline_q)

    # 3) 粗略找所有局部峰（在平滑曲线中）
    # 条件：y[i] 是局部最大且高于基线
    peaks_all = []
    for i in range(1, n - 1):
        if (y_smooth[i] >= y_smooth[i - 1] and
            y_smooth[i] >= y_smooth[i + 1] and
            y_smooth[i] > y_base):
            peaks_all.append(i)

    peaks_all = np.asarray(peaks_all, dtype=int)

    if peaks_all.size == 0:
        raise ValueError("No peaks found above baseline.")

    # 按峰高从大到小排序，取前 k 个
    peaks_sorted_by_height = peaks_all[np.argsort(y_smooth[peaks_all])[::-1]]
    if peaks_sorted_by_height.size < k:
        raise ValueError(
            f"Found only {peaks_sorted_by_height.size} peaks above baseline, "
            f"but k={k}."
        )

    peaks_topk = peaks_sorted_by_height[:k]
    # 为了返回结果更直观：按 x 从小到大排序
    peaks_topk = np.sort(peaks_topk)

    # 4) 对每个峰，找左侧半高点
    x_halfs = np.full(k, -1.0, dtype=float)
    y_half_list = np.zeros(k, dtype=float)

    for idx_peak_idx, k_peak in enumerate(peaks_topk):
        y_peak = y_smooth[k_peak]

        # 当前峰的半高
        y_half = y_base + 0.5 * (y_peak - y_base)
        y_half_list[idx_peak_idx] = y_half

        # 只在 [0, k_peak] 上找“第一次 >= 半高”的位置
        seg = y_smooth[:k_peak + 1]
        # 从 k_peak 往左扫，找第一次从下往上穿越半高的位置
        j_cross = None
        for j in range(k_peak, 0, -1):
            if seg[j] >= y_half and seg[j - 1] < y_half:
                j_cross = j
                break

        if j_cross is None:
            # 理论上不该发生（前面 max/ min 已经判过）
            x_halfs[idx_peak_idx] = -1.0
            continue

        i = j_cross - 1
        x0, x1 = float(i), float(j_cross)
        y0, y1 = seg[i], seg[j_cross]

        if y1 == y0:
            # 平坦段，取中点
            x_half = 0.5 * (x0 + x1)
        else:
            t = (y_half - y0) / (y1 - y0)
            x_half = x0 + t * (x1 - x0)

        x_halfs[idx_peak_idx] = x_half

    info = {
        "y_smooth": y_smooth,
        "y_base": y_base,
        "peaks_idx": peaks_topk,
        "peaks_val": y_smooth[peaks_topk],
        "y_half": y_half_list,
    }

    return x_halfs, info

def bins_count_image_from_xy(pos, W=512, H=512, round_mode="truncate", detid=0):
    """
    对一组[x,y]坐标做分箱计数，返回HxW整型图。
    round_mode:
      - 'truncate': 直接astype(int)，向0截断
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
    lin = xi * W + yi
    img = np.bincount(lin, minlength=W * H).reshape(H, W)
    return img

def process_one_txt(txt_path, detid, visible=False):
    if detid == 0:
        data = np.loadtxt(txt_path, delimiter=" ", encoding="utf-8")[:, 6:]
        k = 1
    elif detid == 1:
        data = np.loadtxt(txt_path, delimiter=" ", encoding="utf-8")[:, 2:6]
        k = 2
    data = data.astype(np.float64)
    adc_sums = np.sum(data, axis=1)
    adc_sums_nozero = adc_sums.copy()
    adc_sums_nozero[adc_sums_nozero == 0] = np.finfo(float).eps
    x_coords = np.round((data[:,0] + data[:,1] - data[:,2] - data[:,3]) / adc_sums_nozero * 200) + 255
    y_coords = np.round((data[:,0] - data[:,1] - data[:,2] + data[:,3]) / adc_sums_nozero * 200) + 255
    # 创建一个逻辑掩码，标记所有位置坐标在 [1, 512] 范围内的有效事件
    valid_pos_mask = (x_coords >= 0) & (x_coords < 512) & (y_coords >= 0) & (y_coords < 512)

    valid_x = (511.0 - x_coords[valid_pos_mask]).astype(int)
    if detid == 1:
        valid_y = (511.0 - y_coords[valid_pos_mask]).astype(int)
    else:
        valid_y = y_coords[valid_pos_mask].astype(int)
    img = bins_count_image_from_xy(np.column_stack((valid_x, valid_y)), detid=detid)
    x_half, info = findcols(np.sum(img, axis=0), k=k)
    
    if visible:
        col_smooth, col_base, peaks_idx = info["y_smooth"], info["y_base"], info["peaks_idx"]
        plt.imshow(img, cmap='gray', aspect='auto')
        # plt.plot(np.arange(col_smooth.shape[0]), col_smooth)
        # plt.axhline(y=col_base)
        # for i in range(x_half.shape[0]):
        #     plt.axvline(x=x_half[i])
        #     plt.axvline(x=peaks_idx[i], color='red')
        plt.show()
    return x_half

if __name__ == "__main__":

    # xhalf = process_one_txt("./TrueData/new_calibration/2025_11_17_14_23_32/Energy.txt", detid=0)
    # print(xhalf)
    
    # id {0: 单缝, 1: 双缝}
    base_dir = Path(r"C:\Users\46595\Learning\BackScatter\code\FlyPoint\TrueData\new_calibration")
    outputpath0 = f"./data/Calibration_data/cols_calibration0.npy"
    outputpath1 = f"./data/Calibration_data/cols_calibration1.npy"

    # 只保留一级子目录，并按名字排序（就是你 tree 输出来的顺序）
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    subdirs = sorted(subdirs, key=lambda p: p.name)
    cols0, cols1 = [], []
    for folder in subdirs:
        print("进入文件夹:", folder.name)
        # 找出该文件夹下所有 txt 文件（不递归）
        for fname in os.listdir(folder):
            if fname.lower().endswith(".txt"):
                txt_path = folder / fname
                # col0 = process_one_txt(txt_path, detid=0, visible=True)
                col1 = process_one_txt(txt_path, detid=1, visible=True)
                # cols0.append(col0)
                cols1.append(col1)
    np.save(outputpath0, np.array(cols0))
    np.save(outputpath1, np.array(cols1))
    