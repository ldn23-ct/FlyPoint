import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

def optimal_processing_chunk_len(h5_path, dataset_key, bytes_budget=256*1024*1024):
    '''
    确定每次读取时的最佳大小
    '''
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        dtype_size = np.dtype(dset.dtype).itemsize
        h5_chunk = dset.chunks[0] if dset.chunks else None

    if h5_chunk is None:
        # contiguous 存储，无需对齐
        chunk_len = bytes_budget // dtype_size
        print(f"dataset is contiguous; suggested chunk_len ≈ {chunk_len:,}")
        return chunk_len

    # 以 HDF5 chunk 为单位取整
    n_per_chunk = h5_chunk
    max_chunks = bytes_budget // (n_per_chunk * dtype_size)
    chunk_len = max(1, max_chunks) * n_per_chunk

    print(f"---------------------------------评估读取存储块大小---------------------------------")
    print(f"HDF5 chunks = {n_per_chunk}, dtype size = {dtype_size} bytes")
    print(f"→ 每存储块约 {n_per_chunk*dtype_size/1024:.1f} KB")
    print(f"→ 每次读取 {max_chunks} 个存储块，共 {chunk_len:,} 个元素")
    print(f"→ 约 {(chunk_len*dtype_size)/(1024**2):.1f} MB 数据")
    print(f"---------------------------------评估读取存储块大小---------------------------------")
    return chunk_len

def pass_min_max_h5(h5_path, dataset_key, bytes_budget=256*1024*1024):
    '''
    返回数据集中的最小/大值，数据长度以及类型
    '''
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        assert dset.ndim == 1, "期望一维时间戳"
        dtype = dset.dtype
        chunk_len = optimal_processing_chunk_len(h5_path, dataset_key, bytes_budget)

        n = dset.shape[0]
        tmin = None
        tmax = None
        for s in range(0, n, chunk_len):
            e = min(s + chunk_len, n)
            # 这一步只把所需切片拉到内存（懒加载）
            arr = dset[s:e]
            cmin = arr.min()
            cmax = arr.max()
            tmin = cmin if tmin is None else min(tmin, cmin)
            tmax = cmax if tmax is None else max(tmax, cmax)
        return tmin, tmax, n, dtype, chunk_len

def Time_Counts(h5_path, dataset_key,
                bin_width, t0=None, t1=None, time_step=8,
                bytes_budget=256*1024*1024,
                right_inclusive=False):
    '''
    对事件信息进行分箱计数
    
    Args:
        bin_width: 带宽，一个箱涵盖的事件范围。
        
    Returns:
        edges: 箱的左右沿，用daq本身的时间戳来表示。 (n+1,)  \\
        counts: 每个箱的计数。 (n,)
    '''
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        chunk_len = 1
        assert dset.ndim == 1, "期望一维时间戳"
        # 第一遍：确定范围
        if t0 is None or t1 is None:
            _t0, _t1, n, dtype, chunk_len = pass_min_max_h5(h5_path, dataset_key, bytes_budget)
            if t0 is None: t0 = _t0
            if t1 is None: t1 = _t1
        else:
            n = dset.shape[0]

        # 计算 bin 数
        assert bin_width > 0
        span = max(0, t1 - t0)
        nbins = int(np.ceil((span + (1 if right_inclusive else 0)) / bin_width))
        if nbins <= 0:
            nbins = 1
            
        counts = np.zeros(nbins, dtype=np.int64)
        # 第二遍：分块计数
        for s in range(0, n, chunk_len):
            e = min(s + chunk_len, n)
            chunk = dset[s:e]                      # 懒加载，只读到内存这部分
            idx = (chunk - t0) // bin_width        # 整型除法，避免浮点
            np.clip(idx, 0, nbins-1, out=idx)      # 边界裁剪
            bc = np.bincount(idx, minlength=nbins) # 块内快速计数
            counts += bc
        edges = t0  + np.arange(nbins+1, dtype=np.int64) * bin_width
    return edges, counts

def TimeClassification(edges, cnts,
                       angle_v, angle_a,
                       time_width=1e5, time_step=8,
                       W=5, k=3):
    '''
    利用计数信息找到每个周期的上升沿与下降沿
    
    Args:
        edges: 分箱计数中的边沿
        cnts: 分箱计数中的计数
        angle_v: 角速度
        angle_a: 角加速度，需要排除加减速区间
        time_width: 分箱代表的时间长度，默认0.1ms
        time_step: daq每次采样的时间间隔，默认8ns
        W: 卷积过滤信号中的卷积长度
        k: 判断是否上升/下降的阈值
    
    Returns:
        segs: [n, 2] 返回每个周期的起始、终止点, 起始、终止点均用时间戳表示
    '''
    acceleration_time = angle_v / angle_a * 1e9
    T = 18 / angle_v * 1e9
    start_idx = (acceleration_time + 1e9) / time_width
    end_idx = cnts.shape[0] -  (acceleration_time / time_width)
    cnts_threshold = int(np.max(cnts[:int(3*T)]) * 0.1)
    cnts_filter = (cnts >= cnts_threshold).astype(int)
    cnts_filter = np.convolve(cnts_filter, np.ones(W, dtype=int), mode='same')
    cnts_filter = (cnts_filter >= k).astype(int)
    diff = np.diff(cnts_filter)
    rising_idx  = np.flatnonzero(diff == +1) + 1
    falling_idx = np.flatnonzero(diff == -1)
    segs = []
    i = 0
    j = 0
    while i < len(rising_idx) and j < len(falling_idx):
        if rising_idx[i] < falling_idx[j]:
            segs.append((rising_idx[i], falling_idx[j]))
            i += 1; j += 1
        else:
            j += 1  # 丢弃不合法的 end
    segs = np.array(segs)
    mask = (segs[:, 0] >= start_idx-200) & (segs[:, 1] <= end_idx)
    segs = segs[mask].astype(int)
    
    #  若使用plot_slice_with_segments请注释以下代码
    # for i in range(segs.shape[0]):
    #     s, e = segs[i, 0], segs[i, 1]
    #     segs[i, 0] = edges[s]
    #     segs[i, 1] = edges[e+1]

    return np.array(segs)

def plot_slice_with_segments(x, segs, a, b, annotate=False):
    """
    x: 计数信息
    segs: list of (start, end)
    a, b: 可视化的切片区间，使用 x[a:b]（半开区间）
    """
    x = np.asarray(x)
    assert 0 <= a < b <= len(x), "切片越界"

    # 片段数据与横轴（局部坐标）
    xs = x[a:b]
    t_local = np.arange(b - a)

    fig, ax = plt.subplots()
    ax.scatter(t_local, xs, s=8)  # 纯散点
    
    mask = (segs[:, 0] >= a) & (segs[:, 1] <= b)
    segs = segs[mask]
    # 处理每个周期段与切片的交集，并画 start/end 的垂直线
    for (s, e) in segs:
        # 与 [a, b-1] 是否相交
        if e < a or s > b - 1:
            continue
        # 映射到切片局部坐标
        s_loc = s - a
        e_loc = e - a

        ax.axvline(s_loc, linestyle='--', color='r')
        ax.axvline(e_loc, linestyle='--')

        if annotate:
            ax.text(s_loc, np.nanmax(xs), f'start={s}', rotation=90, va='bottom', ha='right')
            ax.text(e_loc, np.nanmax(xs), f'end={e}', rotation=90, va='bottom', ha='left')

    ax.set_xlabel(f'index (local in [{a}:{b}))')
    ax.set_ylabel('x')
    ax.set_title(f'x[{a}:{b}] with segment boundaries')
    plt.tight_layout()
    plt.show()

def load_xy_groups(csv_path, i, j):
    """
    读入CSV，仅保留 x, y, angle_degree，按角度0..15分组。
    返回:
      groups: dict[int -> (n,2) numpy数组]，列顺序 [y, x]（方便后续计数）
    """
    df = pd.read_csv(csv_path, usecols=["x", "y", "angle_degree", "period_id"])

    # 清理NaN
    df = df.dropna(subset=["x", "y", "angle_degree", "period_id"])

    # period_id 天然有序 -> 直接按区间筛选（不开排序）
    pid = pd.to_numeric(df["period_id"], errors="coerce")
    mask = (pid >= i) & (pid < j)
    df = df[mask]
    groups = {a: np.empty((0, 2), dtype=float) for a in range(16)}
    if df.empty:
        print("所在序列为空")
        return groups

    # 角度转数值并去掉不可转的
    ang = pd.to_numeric(df["angle_degree"], errors="coerce")
    df = df[ang.notna()]
    df["angle_degree"] = ang.astype(np.int64)

    # 仅保留 0..15
    df = df[(df["angle_degree"] >= 0) & (df["angle_degree"] <= 15)]

    if df.empty:
        return groups
    
    for a, g in df.groupby("angle_degree", sort=True):
        arr_yx = g[["x", "y"]].to_numpy(copy=False)
        groups[int(a)] = arr_yx

    return groups

def bins_count_image_from_yx(pos, W=512, H=512, round_mode="truncate"):
    """
    对一组[y,x]坐标做分箱计数，返回HxW整型图。
    round_mode:
      - 'truncate': 直接astype(int)，向0截断（与你原代码一致）
      - 'round':    np.rint 四舍五入
      - 'floor':    np.floor 向下取整
    """
    if pos.size == 0:
        return np.zeros((H, W), dtype=np.int64)

    x = pos[:, 1]
    y = pos[:, 0]

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
    return img[:, ::-1]


if __name__ == "__main__":
    # h5_path = "./TrueData/tasks/2025-10-29_16-09_5f844ddc/data/original_data.h5"
    # dataset_key = "daq0/timestamps"
    # time_width = 1e5  # 0.1ms
    # time_step = 8  # daq采样频率8ns
    # bin_width = int(time_width/ time_step)
    # edges, counts = Time_Counts(h5_path, dataset_key, bin_width)
    # segs = TimeClassification(edges, counts, 1200, 200)  #角速度1200deg/s，角加速度200deg/s^2
    # plot_slice_with_segments(counts, segs, 70000, 73000)

    csv_path = './TrueData/tasks/2025-10-29_16-09_5f844ddc/data/events_with_angle.csv'
    xy_groups = load_xy_groups(csv_path, 0, 200)
    fig, axes = plt.subplots(4, 8, figsize=(18, 9), constrained_layout=True)
    for i in range(16):
        img = bins_count_image_from_yx(xy_groups[i])
        img_sum = np.sum(img, axis=0)
        axes[int(i/4), int(i%4)].imshow(img, cmap='gray', aspect='equal')
        axes[int(i/4), int(i%4+4)].plot(np.arange(img_sum.shape[0]), img_sum)
    # plt.tight_layout()
    plt.show()