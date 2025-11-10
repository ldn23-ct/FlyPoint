import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from copy import deepcopy
import h5py

class RailMotionParser:
    """
    导轨运动记录解析与修正
    Attributes:
        rail_acc_dec (float): 导轨加速度与减速度值
        rail_long_axis_min (float): 导轨长轴有效最小值
        rail_long_axis_max (float): 导轨长轴有效最大值
        rail_uniform_margin (float): 导轨匀速阶段相较于长轴有效边界的余量
        rail_long_axis (str): 导轨长轴方向 "x" 或 "y"
        rail_timestamp_offset (float): 导轨时间戳偏移量
        daq_start_collecting_time (float): DAQ开始采集时间
    """
    def __init__(self,
                 rail_acc_dec,
                 rail_long_axis_min, rail_long_axis_max, rail_uniform_margin = 0.5,
                 rail_long_axis: str = "x",
                 rail_timestamp_offset: float = 0.0,
                 daq_start_collecting_time: float = 0.0):
        self.rail_acc_dec = rail_acc_dec
        self.rail_long_axis_min = rail_long_axis_min
        self.rail_long_axis_max = rail_long_axis_max
        self.rail_uniform_margin = rail_uniform_margin
        self.rail_long_axis = rail_long_axis
        self.rail_timestamp_offset = rail_timestamp_offset
        self.daq_start_collecting_time = daq_start_collecting_time
    
    def correct_rail_motions(self, motions):
        """
        校正导轨运动记录的时间戳为以上位机DAQ开始采集时间为零点，单位为s
        """
        motions[:, 0] *= 1e-9  # 时间单位从ns转为s
        # 校正为DAQ的超始时间
        motions[:, 0] += self.rail_timestamp_offset - self.daq_start_collecting_time
        return motions
    
    def refine_rail_motions(self, motions):
        """
        修正导轨运动记录，插入减速或加速前后匀速运动阶段的边界记录
        """
        return self.insert_bound_motions(
            motions,
            self.rail_acc_dec,
            self.rail_long_axis_min, self.rail_long_axis_max, self.rail_uniform_margin,
            self.rail_long_axis,
            vel_tol=1e-1
        )

    @staticmethod
    def insert_bound_motions(
        motions, 
        acc_dec, 
        valid_long_axis_min, valid_long_axis_max,
        uniform_margin: float = 0.5,
        long_axis: str = "x",
        vel_tol: float = 1e-2
    ):
        """
        插入匀速运动阶段，有效边界的运动记录

        Args:
            motions (list): 新的运动记录列表，每一个记录为[timestamp, x, y, vx, vy, ax, ay]的列表
            valid_long_axis_min (float): 长轴的有效最小值
            valid_long_axis_max (float): 长轴的有效最大值
            uniform_margin (float, optional): 匀速阶段相较于长轴有效边界的余量. Defaults to 0.5.
            long_axis (str, optional): 长轴. Defaults to "x".
            vel_tol (float, optional): 判断速度为0的容忍度. Defaults to 1e-2.

        Returns:
            list: 修正后的运动记录列表  [[timestamp, x, y, vx, vy, ax, ay]]
        """
        if len(motions) < 2:
            return motions
        long_axis_idx = 1 if long_axis == "x" else 2
        short_axis_idx = 2 if long_axis == "x" else 1
        # 用于插入的匀速坐标
        inserted_long_axis_min_bound = valid_long_axis_min - uniform_margin
        inserted_long_axis_max_bound = valid_long_axis_max + uniform_margin
        
        out_motions = [motions[0]]
        for i in range(1, len(motions)):
            cur_motion = motions[i]
            last_motion = out_motions[-1]
            same_long_axis = abs(cur_motion[short_axis_idx] - last_motion[short_axis_idx]) < 1e-3
            if same_long_axis:
                last_is_uniform = abs(last_motion[long_axis_idx + 2]) > vel_tol and last_motion[long_axis_idx + 4] == 0
                cur_is_uniform = abs(cur_motion[long_axis_idx + 2]) > vel_tol and cur_motion[long_axis_idx + 4] == 0
                
                pos_dir = np.sign(cur_motion[long_axis_idx] - last_motion[long_axis_idx])
                insert_long_axis_end_bound = inserted_long_axis_max_bound if pos_dir > 0 else inserted_long_axis_min_bound
                insert_long_axis_start_bound = inserted_long_axis_min_bound if pos_dir > 0 else inserted_long_axis_max_bound
                if last_is_uniform and not cur_is_uniform:
                    # 前一个记录为匀速，后一个记录不是，为减速过程，需要插入有效边界的记录
                    # 若前一个匀速记录已经位于轴有效边界之外，不需要插入记录
                    if inserted_long_axis_min_bound < last_motion[long_axis_idx] < inserted_long_axis_max_bound:
                        uniform_speed = abs(last_motion[long_axis_idx + 2])
                        ds = abs(insert_long_axis_end_bound - last_motion[long_axis_idx])
                        dt = ds / uniform_speed
                        inserted_motions = deepcopy(last_motion)
                        inserted_motions[0] = last_motion[0] + dt
                        if inserted_motions[0] > cur_motion[0]:
                            inserted_motions[0] = cur_motion[0]
                        inserted_motions[long_axis_idx] = insert_long_axis_end_bound
                        inserted_motions[long_axis_idx + 4] = cur_motion[long_axis_idx + 4] or acc_dec
                        out_motions.append(inserted_motions)
                elif cur_is_uniform and not last_is_uniform:
                    # 前一个记录不是匀速，当前记录为匀速，为加速过程，需要插入有效边界的记录
                    # 如果匀速记录在长轴边界之外，不需要插值
                    if inserted_long_axis_min_bound < cur_motion[long_axis_idx] < inserted_long_axis_max_bound:
                        uniform_speed = abs(cur_motion[long_axis_idx + 2])
                        ds = abs(insert_long_axis_start_bound - cur_motion[long_axis_idx])
                        dt = ds / uniform_speed
                        inserted_motions = deepcopy(cur_motion)
                        inserted_motions[0] = cur_motion[0] - dt
                        if inserted_motions[0] < last_motion[0]:
                            inserted_motions[0] = last_motion[0]
                        inserted_motions[long_axis_idx] = insert_long_axis_start_bound
                        inserted_motions[long_axis_idx + 4] = last_motion[long_axis_idx + 4] or acc_dec
                        out_motions.append(inserted_motions)
            out_motions.append(cur_motion)
        return out_motions

    @staticmethod
    def get_count_events_with_coords(
        event_ts,
        rail_motions,
    ):
        """
        获取DAQ计数事件对应时间戳的导轨坐标
        Args:
            event_ts: 计数事件的时间戳, 单位为ns
            rail_motions: 导轨运动记录 [[timestamp, x, y, vx, vy, ax, ay]], 其中timestamp单位为ns
        Returns:
            event_rail_xy: 有效计数事件对应的导轨坐标 [[x, y]]  Mx2
            valid_event_ts: 有效的计数事件时间戳 [[timestamp]]  Mx1
            event_mask: 有效的计数事件掩码
        """
        rail_ts_min = np.min(rail_motions[:, 0])
        rail_ts_max = np.max(rail_motions[:, 0])
        
        valid_ts_min = rail_ts_min
        valid_ts_max = rail_ts_max

        event_mask = (event_ts >= valid_ts_min) & (event_ts <= valid_ts_max)
        valid_event_ts = event_ts[event_mask]
        
        event_rail_x = np.interp(valid_event_ts, rail_motions[:, 0], rail_motions[:, 1]).reshape(-1, 1)
        event_rail_y = np.interp(valid_event_ts, rail_motions[:, 0], rail_motions[:, 2]).reshape(-1, 1)
        event_rail_xy = np.hstack([event_rail_x, event_rail_y])

        return event_rail_xy, valid_event_ts.reshape(-1, 1), event_mask

class TimeClass:
    def __init__(self,
                time_width,  # 0.1ms
                time_step,     # daq采样频率8ns
                angle_v,    # 角速度1200deg/s
                angle_a,     # 角加速度200deg/s^2
    ):
        self.time_width = time_width
        self.time_step = time_step
        self.bin_width = int(time_width / time_step)
        self.angle_v = angle_v
        self.angle_a = angle_a

    def Time_Counts(self, timestamps, bin_width=None, t0=None, t1=None, right_inclusive=False):
        '''
        对事件信息进行分箱计数。
        
        Args:
            timestamps: 包含所有事件时间戳的一维 NumPy 数组。
            bin_width: 带宽，一个箱涵盖的事件范围。
            t0, t1: 计数的起始和结束时间。如果为 None，则从数据中自动确定。
            right_inclusive: bin 的右边界是否包含在内。
            
        Returns:
            edges: 箱的左右沿，用 daq 本身的时间戳来表示。 (n+1,)
            counts: 每个箱的计数。 (n,)
        '''
        if bin_width == None:
            bin_width = self.bin_width
        
        assert timestamps.ndim == 1, "期望一维时间戳数组"

        # 第一步：确定范围
        if t0 is None:
            t0 = timestamps.min()
        if t1 is None:
            t1 = timestamps.max()

        # 计算 bin 数
        assert bin_width > 0
        span = max(0, t1 - t0)
        nbins = int(np.ceil((span + (1 if right_inclusive else 0)) / bin_width))
        if nbins <= 0:
            nbins = 1
            
        # 直接对整个数组进行分箱和计数
        # 注意：这里假设整个时间戳数组可以被加载到内存中
        idx = ((timestamps - t0) // bin_width).astype(int)
        
        # 过滤掉在 [t0, t1] 范围之外的索引
        valid_mask = (idx >= 0) & (idx < nbins)
        
        # 使用 bincount 进行高效计数
        counts = np.bincount(idx[valid_mask], minlength=nbins)
        
        edges = t0 + np.arange(nbins + 1, dtype=np.int64) * bin_width
        
        return edges, counts

    def TimeClassification(self, edges, cnts,
                        angle_v=None, angle_a=None,
                        time_width=None, time_step=None,
                        W=5, k=3):
        if angle_v == None:
            angle_v = self.angle_v
        if angle_a == None:
            angle_a = self.angle_a
        if time_width == None:
            time_width = self.time_width
        if time_step == None:
            time_step = self.time_step
            
        acceleration_time = angle_v / angle_a * 1e9
        T = 18 / angle_v * 1e9
        start_idx = (acceleration_time + 1e9) / time_width
        end_idx = cnts.shape[0] -  (acceleration_time / time_width)
        
        # --- 修正阈值计算 ---
        slice_end_idx = int(3 * T / time_width)
        slice_end_idx = min(slice_end_idx, len(cnts))
        initial_counts = cnts[:slice_end_idx]
        if initial_counts.size > 0:
            cnts_threshold = int(np.max(initial_counts) * 0.1)
        else:
            cnts_threshold = int(np.mean(cnts) * 0.1)
        cnts_threshold = max(cnts_threshold, 1)
        # --- 阈值计算结束 ---

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
                j += 1
        segs = np.array(segs)

        if segs.size == 0:
            return np.array([]).reshape(0, 2)

        mask = (segs[:, 0] >= start_idx-200) & (segs[:, 1] <= end_idx)
        segs = segs[mask].astype(int)
        
        # --- 修改：取消注释，确保返回的是时间戳 ---
        # 这个循环将索引转换为实际的时间戳
        for i in range(segs.shape[0]):
            s, e = segs[i, 0], segs[i, 1]
            segs[i, 0] = edges[s]
            segs[i, 1] = edges[e+1]

        return np.array(segs)

    def segment_by_angle(self, data, segs, angle_v=None, time_step=None):
        """
        将每个周期内的事件按角度划分。

        Args:
            data (np.array): 完整的事件数据 (N, 4)，包含 [timestamp, x, y, energy]。
            segs (np.array): 周期起止点数组 (M, 2)，包含 [start_timestamp, end_timestamp]。
            angle_v (float): 角速度 (度/秒)。

        Returns:
            pd.DataFrame: 一个包含所有周期内事件的 DataFrame，
                        新增了 'period_id' 和 'angle' 两列。
        """
        if angle_v == None:
            angle_v = self.angle_v
        if time_step == None:
            time_step = self.time_step
        
        # 将角速度从 度/秒 转换为 度/纳秒
        angle_v_ns = angle_v / 1e9

        # 创建一个包含所有周期内事件的列表
        all_period_events = []

        print("正在按角度划分周期...")
        for i, (start_ts, end_ts) in enumerate(segs):
            # 1. 筛选出当前周期内的所有事件
            period_mask = (data[:, 0] >= start_ts) & (data[:, 0] < end_ts)
            period_data = data[period_mask]

            if period_data.shape[0] == 0:
                continue # 如果这个周期没有事件，则跳过

            # 2. 计算每个事件相对于周期开始的时间差（纳秒）
            time_elapsed = (period_data[:, 0] - start_ts) * time_step

            # 3. 计算每个事件对应的角度
            angles = time_elapsed * angle_v_ns
            
            # 4. 将角度信息和周期ID添加到数据中
            # 创建一个 DataFrame 以方便操作
            df = pd.DataFrame(period_data, columns=['timestamp', 'x', 'y', 'energy', 'rail_x', 'rail_y'])
            df['period_id'] = i
            df['angle'] = angles
            
            all_period_events.append(df)
            
            if (i + 1) % 100 == 0: # 每处理100个周期打印一次进度
                print(f"  已处理 {i+1}/{len(segs)} 个周期...")

        if not all_period_events:
            print("警告：在所有识别出的周期内均未找到任何事件。")
            return pd.DataFrame(columns=['timestamp', 'x', 'y', 'energy', 'period_id', 'angle'])

        # 将所有周期的 DataFrame 合并为一个
        final_df = pd.concat(all_period_events, ignore_index=True)
        
        # 为了方便后续按1度区间筛选，我们可以增加一个整数角度列
        final_df['angle_degree'] = final_df['angle'].astype(int)
        
        return final_df

    def SaveCSV(self, data, filepath):
        timestamps = data[:, 0]
        
        print("正在对时间戳进行分箱计数...")
        edges, counts = self.Time_Counts(timestamps)
        
        print("正在进行时间分类...")
        segs = self.TimeClassification(edges, counts)
        
        print(f"找到了 {len(segs)} 个周期。")
        
        # --- 新增处理流程 ---
        if segs.size > 0:
            # 调用新函数按角度划分
            segmented_df = self.segment_by_angle(data, segs)
            
            print(f"已将 {len(segmented_df)} 个事件划分到 {len(segs)} 个周期中。")
            
            # 保存带有角度信息的数据到 CSV 文件
            output_csv_path = filepath
            print(f"正在保存结果到 {output_csv_path}...")
            segmented_df.to_csv(output_csv_path, index=False)
            print("保存完成。")
            
            # --- 示例：如何使用划分好的数据 ---
            # 筛选出第 5 度角的所有事件
            angle_5_events = segmented_df[segmented_df['angle_degree'] == 5]
            print(f"\n示例：在第 5 度角区间内，共找到了 {len(angle_5_events)} 个事件。")
            
            # 筛选出第 10 个周期的所有事件
            period_10_events = segmented_df[segmented_df['period_id'] == 10]
            print(f"示例：在第 10 个周期内，共找到了 {len(period_10_events)} 个事件。")

        else:
            print("未找到任何周期，无法进行角度划分。")    

    def plot_slice_with_segments(self, x, segs, a, b, annotate=False):
    # ...existing code...
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


if __name__ == "__main__":
    dir = "./TrueData/tasks/2025-10-29_16-26_e3c8fa25/data/"
    npy_path = dir + "position.npy"
    motion_file = dir + "motion.h5"
    daq_origin_data_file = dir + "original_data.h5"
    outputpath = dir + "events_with_angle.csv"
    print(f"从 {npy_path} 加载解码数据...")
    print(f"从 {motion_file} 加载导轨数据...")
    print(f"从 {daq_origin_data_file} 加载daq数据...")
    
    data = np.load(npy_path)
    with h5py.File(motion_file, "r") as f:
        motions = f["rail"][:]
        rail_timestamp_offset = f.attrs["rail_timestamp_offset"]
    with h5py.File(daq_origin_data_file, "r") as f:
        daq0 = f["daq0"]
        start_collecting_time = daq0.attrs["start_collecting_time"]
    
    # 移除时间戳或位置为 NaN 的行，因为它们无法用于后续分析
    valid_data_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_data_mask]
    event_ts = data[:, 0].reshape(-1, 1)
    
    print(f"加载了 {len(event_ts):,} 个有效事件。")
    
    parser = RailMotionParser(
        rail_acc_dec=2000,
        rail_long_axis_min=-100, rail_long_axis_max=0.0,
        rail_uniform_margin=0.5,
        rail_long_axis="x",
        rail_timestamp_offset=rail_timestamp_offset,
        daq_start_collecting_time=start_collecting_time
    )
    
    motions = parser.correct_rail_motions(motions)
    motions = parser.refine_rail_motions(motions)
    motions = np.array(motions)
    motions[:, 0] *= 1e9/8  # 时间单位转回8ns
    # event_ts *= 8 # 时间单位从8ns转回ns
    
    valid_event_rail_xy, valid_event_ts, event_mask = parser.get_count_events_with_coords(event_ts, motions)
    # 获取有效数据
    data = data[event_mask[:, 0]]
    data = np.hstack([data, valid_event_rail_xy])

    # 定义分箱参数
    time_width = 1e5  # 0.1ms
    time_step = 8     # daq采样频率8ns
    bin_width = int(time_width / time_step)
    angle_v = 1200    # 角速度1200deg/s
    angle_a = 200     # 角加速度200deg/s^2
    
    timeclass = TimeClass(time_width, time_step, angle_v, angle_a)
    timeclass.SaveCSV(data, outputpath)
    
    # 可视化部分可以保持不变，用于调试
    # print("显示其中一个周期的计数图...")
    # plot_slice_with_segments(counts, segs, 70000, 73000)