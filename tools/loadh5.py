import numpy as np
import h5py
import matplotlib.pyplot as plt

def list_h5_attrs(h5_path): 
    with h5py.File(h5_path, 'r') as f: 
        def print_attrs(name, obj): 
            if len(obj.attrs) > 0: 
                print(f"[{name}] has attributes:") 
                for k, v in obj.attrs.items(): 
                    print(f" {k}: {v}") 
        f.visititems(print_attrs)

def print_structure(name, obj):
    '''
    查询h5文件结构
    
    Example:
        with h5py.File('data.h5', 'r') as f:
            f.visititems(print_structure)
    '''
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name} shape={obj.shape}, dtype={obj.dtype}")

def inspect_h5(h5_path, dataset_key):
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        n = dset.shape[0]
        print(f"dataset: {dataset_key}")
        print(f"  shape: {dset.shape}, dtype: {dset.dtype}")
        print(f"  compression: {dset.compression}, shuffle: {dset.shuffle}, fletcher32: {dset.fletcher32}")
        print(f"  chunks (HDF5 storage layout): {dset.chunks}")  # None 表示 contiguous
        return n, str(dset.dtype)
    
def optimal_processing_chunk_len(h5_path, dataset_key, bytes_budget=256*1024*1024):
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
        return tmin, tmax, n, dtype

def Time_Counts(h5_path, dataset_key,
                bin_width, t0=None, t1=None, time_step=8,
                bytes_budget=256*1024*1024,
                right_inclusive=False):
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        assert dset.ndim == 1, "期望一维时间戳"
        print(dset[0:10])
        # 第一遍：确定范围
        if t0 is None or t1 is None:
            _t0, _t1, n, dtype = pass_min_max_h5(h5_path, dataset_key, bytes_budget)
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
        chunk_len = optimal_processing_chunk_len(h5_path, dataset_key, bytes_budget)
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

def fs(h5_path, dataset_key, bytes_budget=256*1024*1024):
    with h5py.File(h5_path, 'r') as f:
        t_off = f.attrs['fs_timestamp_offset']
        dset = f[dataset_key]
        time = dset[:, 0]
        angle = dset[:, 1]
    return t_off, time, angle

def hits_to_image_integer(coords, W=512, H=512, clip=True):
    """
    coords: shape (n,2) -> [:,0]=x_idx, [:,1]=y_idx，整型像素坐标
    """
    yi = coords[:,0].astype(np.int64, copy=False)
    xi = coords[:,1].astype(np.int64, copy=False)

    if clip:
        m = (xi>=0) & (xi<W) & (yi>=0) & (yi<H)
        xi = xi[m]; yi = yi[m]
    else:
        if (xi.min()<0) or (xi.max()>=W) or (yi.min()<0) or (yi.max()>=H):
            raise ValueError("像素索引越界，考虑 clip=True 或修正坐标。")

    lin = yi * W + xi
    img = np.bincount(lin, minlength=W*H).reshape(H, W)

    plt.figure(figsize=(6,6))
    plt.imshow(img, origin='lower', cmap='gray', aspect='equal')
    plt.xlabel('x pixel'); plt.ylabel('y pixel'); plt.title('Detector hits (counts per pixel)')
    plt.colorbar(label='counts')
    plt.show()
    return img  

def TimeClassification(edges, cnts,
                       angle_v, angle_a,
                       time_width=1e5, time_step=8,
                       W=5, k=3):
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
    for i in range(segs.shape[0]):
        s, e = segs[i, 0], segs[i, 1]
        segs[i, 0] = edges[s]
        segs[i, 1] = edges[e+1]
    # N = int(T / time_width)
    # single_sum = np.zeros(N)
    # for i in range(segs.shape[0]):
    #     s, e = segs[i, 0], segs[i, 1]
    #     for j in range(e - s + 1):
    #         single_sum[j] += cnts[j+s]
    # plt.scatter(np.arange(N), single_sum)
    # plt.show()
    
    
    return np.array(segs)

def plot_slice_with_segments(x, segs, a, b, use_open_interval=False, annotate=False):
    """
    x: 1D array
    segs: list of (start, end)，默认 end 为闭区间。如果你的 segs 是 [start, end) 开区间，请设 use_open_interval=True
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
    print(segs)
    # 处理每个周期段与切片的交集，并画 start/end 的垂直线
    for (s, e) in segs:
        if use_open_interval:
            e = e - 1  # 若输入是 [s, e)，为画线方便转为闭区间 [s, e-1]

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

h5_path0 = "./TrueData/tasks/2025-10-29_16-09_5f844ddc/data/original_data.h5"
h5_path1 = "./TrueData/tasks/2025-10-29_16-12_f80c9564/data/original_data.h5"
dataset_key = "daq0/timestamps"
time_width = 1e5  # 0.1ms
time_step = 8  # daq采样频率8ns
bin_width = int(time_width/ time_step)
# edges0, counts0 = Time_Counts(h5_path0, dataset_key, bin_width)
# edges1, counts1 = Time_Counts(h5_path1, dataset_key, bin_width)
# segs = TimeClassification(edges0, counts0, 1200, 200)
# plot_slice_with_segments(counts0, segs, 10500, 12000)

motion_path = "./TrueData/tasks/2025-10-29_16-09_5f844ddc/data/motion.h5"
motion_key = "fs"
# t_off, time, angle = fs(motion_path, motion_key)
# t = time[0:10] + np.around(t_off*1e3)

# _t0, _t1, n, dtype = pass_min_max_h5(h5_path0, dataset_key)
# t = np.zeros(n)
# with h5py.File(h5_path0, 'r') as f:
#     dset = f[dataset_key]
#     t = dset - _t0
# print(t[0:10])
pos = np.load("./tools/position.npy")
print(pos.shape)
with h5py.File(h5_path0, 'r') as f:
    dset = f[dataset_key]
    print(dset.shape)
    pos = np.concatenate((pos, dset), axis=1)
print(pos.shape)

