import numpy as np
from pathlib import Path
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

def findcols(
    y,
    k,
    baseline_q=0.10,      # 用分位数估计底部
    smooth_window=8       # 移动平均窗口
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

def bins_count_image_from_xy(pos, W=512, H=512, round_mode="truncate"):
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
    lin = yi * W + xi
    img = np.bincount(lin, minlength=W * H).reshape(H, W)
    return img

def process_one_txt(txt_path, valid_sum_range: tuple = (100, 32000), posdecode=False, matinfo=None, visible=False):
    datatxt = np.loadtxt(txt_path, delimiter=" ", encoding="utf-8")
    # datatxt = np.genfromtxt(txt_path, delimiter=" ", encoding="utf-8", invalid_raise=False)
    datatxt = datatxt[datatxt[:, 0] == 0]
    data0 = datatxt[:, 6:].astype(np.float64)
    data1 = datatxt[:, 2:6].astype(np.float64)
    data_list = [data0, data1]
    img_list, x_half_list, info_list = [], [], []
    for i in range(2):
        data = data_list[i]
        adc_sums = np.sum(data, axis=1)
        valid_mask = (adc_sums > valid_sum_range[0]) & (adc_sums < valid_sum_range[1])
        data = data[valid_mask, :]
        adc_sums = adc_sums[valid_mask]
        x_coords = np.round((data[:,0] + data[:,1] - data[:,2] - data[:,3]) / adc_sums * 200) + 255
        y_coords = np.round((data[:,0] - data[:,1] - data[:,2] + data[:,3]) / adc_sums * 200) + 255
        if posdecode:
            if matinfo == None:
                raise ValueError("File Not Found: matinfo is None")
            matdir, rows, cols = matinfo["matdir"], matinfo["rows"], matinfo["cols"]
            labelmap = loadmat(matdir + f"/labelmap{i}.mat")['labelmap']
            x_coords = np.clip(np.round(x_coords), 0, 511).astype(int)
            y_coords = np.clip(np.round(y_coords), 0, 511).astype(int)
            pixel_id = labelmap[x_coords, y_coords] - 1
            valid_x = cols - 1 - pixel_id // rows
            if i == 1:
                valid_y = rows - 1 - pixel_id % rows
            else: 
                valid_y = pixel_id % rows
              
        else:
            rows, cols = 512, 512
            # 创建一个逻辑掩码，标记所有位置坐标在 [0, 512) 范围内的有效事件
            valid_pos_mask = (x_coords >= 0) & (x_coords < 512) & (y_coords >= 0) & (y_coords < 512)
            valid_x = (511.0 - x_coords[valid_pos_mask]).astype(int)
            if i == 1:
                valid_y = (511.0 - y_coords[valid_pos_mask]).astype(int)
            else:
                valid_y = y_coords[valid_pos_mask].astype(int)
        # img = bins_count_image_from_xy(np.column_stack((valid_x, valid_y)), W=cols, H=rows)
        img = np.zeros((cols, rows))
        np.add.at(img, (valid_x, valid_y), 1)
        x_half, info = findcols(np.sum(img, axis=0), k=i+1)
        img_list.append(img)
        x_half_list.append(x_half)
        info_list.append(info)
    
    if visible:
        for i in range(2):
            img, x_half, info = img_list[i], x_half_list[i], info_list[i]
            col_smooth, col_base, peaks_idx = info["y_smooth"], info["y_base"], info["peaks_idx"]
            plt.figure()
            plt.imshow(img, cmap='gray', aspect='auto')
            # plt.plot(np.arange(col_smooth.shape[0]), col_smooth)
            # plt.axhline(y=col_base)
            # for i in range(x_half.shape[0]):
            #     plt.axvline(x=x_half[i])
            #     plt.axvline(x=peaks_idx[i], color='red')
            plt.show()
            # parent_dir = Path(txt_path).parent
            # filename = (parent_dir / f"fig_{i}.png").as_posix()
            # plt.savefig(filename)
    return x_half_list

if __name__ == "__main__":

    # col0, col1 = process_one_txt("./TrueData/new_calibration/2025_11_20_11_10_48/Energy.txt", visible=True)
    # print(col0, col1)
    # col0, col1 = process_one_txt("./TrueData/new_calibration/2025_11_20_11_12_0/Energy.txt", visible=True)
    # print(col0, col1)
    
    # id {0: 单缝, 1: 双缝}
    base_dir = Path(r"C:\Users\46595\Learning\BackScatter\code\FlyPoint\TrueData\new_calibration")
    outputpath0 = f"./data/Calibration_data/cols_calibration0.txt"
    outputpath1 = f"./data/Calibration_data/cols_calibration1.txt"

    # 只保留一级子目录，并按名字排序（就是你 tree 输出来的顺序）
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    subdirs = sorted(subdirs, key=lambda p: p.name)
    cols0, cols1 = [], []
    matinfo = {"matdir": "./data/Calibration_data",
               "rows": 44,
               "cols": 44}
    # matdir = "./data/Calibration_data"
    for folder in subdirs:
        print("进入文件夹:", folder.name)
        # 找出该文件夹下所有 txt 文件（不递归）
        for fname in os.listdir(folder):
            if fname.lower().endswith(".txt"):
                txt_path = folder / fname
                col0, col1 = process_one_txt(txt_path, posdecode=True, matinfo=matinfo, visible=True)
                cols0.append(col0)
                cols1.append(col1)
    # np.savetxt(outputpath0, np.array(cols0))
    # np.savetxt(outputpath1, np.array(cols1))
