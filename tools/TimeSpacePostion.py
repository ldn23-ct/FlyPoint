import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from copy import deepcopy
import h5py
from typing import List, Tuple

class MotionParser:
    
    def __init__(self,
                 rail_acc_dec, rail_long_axis_speed,
                 rail_long_axis_min, rail_long_axis_max, rail_uniform_margin = 0.5,
                 rail_long_axis: str = "x",
                 rail_timestamp_offset: float = 0.0,
                 fs_timestamp_offset: float = 0.0,
                 daq_start_collecting_time: float = 0.0):
        self.rail_acc_dec = rail_acc_dec
        self.rail_long_axis_speed = rail_long_axis_speed
        self.rail_long_axis_min = rail_long_axis_min
        self.rail_long_axis_max = rail_long_axis_max
        self.rail_uniform_margin = rail_uniform_margin
        self.rail_long_axis = rail_long_axis
        self.rail_timestamp_offset = rail_timestamp_offset
        self.fs_timestamp_offset = fs_timestamp_offset
        self.daq_start_collecting_time = daq_start_collecting_time
    
    def correct_fs_motions(self, motions, offset=0):
        motions[:, 0] *= 1e-3
        motions[:, 1] -= offset
        motions[:, 0] += self.fs_timestamp_offset - self.daq_start_collecting_time
        return motions
    
    def correct_rail_motions(self, motions):
        motions[:, 0] *= 1e-9  # 时间单位从ns转为s
        motions[:, 0] += self.rail_timestamp_offset - self.daq_start_collecting_time
        return motions
    
    def refine_rail_motions(self, motions):
        motions = np.array(motions)
        motions_sort_idx = np.argsort(motions[:, 0])
        motions = motions[motions_sort_idx]
        return self.insert_bound_motions(
            motions,
            self.rail_acc_dec, 
            self.rail_long_axis_speed,
            self.rail_long_axis_min, self.rail_long_axis_max, self.rail_uniform_margin,
            self.rail_long_axis,
            vel_tol=5e-1
        )
        # return motions

    @staticmethod
    def insert_bound_motions(
        motions, 
        acc_dec, 
        long_axis_target_speed,
        valid_long_axis_min, valid_long_axis_max,
        uniform_margin: float = 0.5,
        long_axis: str = "x",
        vel_tol: float = 0.5
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
                # last_is_uniform = abs(last_motion[long_axis_idx + 2]) > vel_tol and last_motion[long_axis_idx + 4] == 0
                # cur_is_uniform = abs(cur_motion[long_axis_idx + 2]) > vel_tol and cur_motion[long_axis_idx + 4] == 0
                
                last_is_uniform = abs(last_motion[long_axis_idx + 2]) > vel_tol and abs(abs(last_motion[long_axis_idx + 2]) - long_axis_target_speed) < vel_tol
                cur_is_uniform = abs(cur_motion[long_axis_idx + 2]) > vel_tol and abs(abs(cur_motion[long_axis_idx + 2]) - long_axis_target_speed) < vel_tol
                
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
        events,
        rail_motions,
        fs_motions,
        fs_t_min=None
    ):
        """
        插值获取DAQ计数事件对应时间戳的导轨坐标与飞点角度
        Args:
            events: 计数事件
            rail_motions: 导轨运动记录 [[timestamp, x, y, vx, vy, ax, ay]]
            fs_motions: 飞点角度记录 [[timestamp, degree, speed]]
        Returns:
            有效计数事件对应的导轨坐标与飞点角度 [[timestamp, x, y, rail_x, rail_y, fs_degree]]
        """
        event_ts = events[:, 0].reshape(-1, 1)

        rail_ts_min = np.min(rail_motions[:, 0])
        rail_ts_max = np.max(rail_motions[:, 0])
        
        if fs_t_min != None:
            fs_ts_min = fs_t_min
        else:
            fs_ts_min = np.min(fs_motions[:, 0])
        fs_ts_max = np.max(fs_motions[:, 0])
        
        valid_ts_min = max(rail_ts_min, fs_ts_min)
        # valid_ts_max = min(rail_ts_max, fs_ts_max)
        valid_ts_max = rail_ts_max
        print(rail_ts_min, fs_ts_min)

        event_mask = (event_ts >= valid_ts_min) & (event_ts <= valid_ts_max)
        valid_event_ts = event_ts[event_mask]
        
        event_rail_x = np.interp(valid_event_ts, rail_motions[:, 0], rail_motions[:, 1]).reshape(-1, 1)
        event_rail_y = np.interp(valid_event_ts, rail_motions[:, 0], rail_motions[:, 2]).reshape(-1, 1)
        # event_fs_degree = np.interp(valid_event_ts, fs_motions[:, 0], fs_motions[:, 1]).reshape(-1, 1)
        event_rail_xy = np.hstack([event_rail_x, event_rail_y])

        valid_events = (valid_event_ts.reshape(-1, 1), events[:, 1:][event_mask[:, 0]], event_rail_xy)
        return np.hstack(valid_events)

    @staticmethod
    def process_fs_angle(events, spot_step=18):
        """
        处理插值后的飞点孔
        Args:
            events: 事件列表 [[timestamp, x, y, rail_x, rail_y, fs_degree]]
            spot_step: 飞点孔的步长，单位为度
        Returns:
            处理后的事件列表 [[timestamp, x, y, rail_x, rail_y, fs_degree]] fs_degree在[-spot_step / 2, spot_step / 2]之间
        """
        fs_angle = events[:, 5]
        fs_angle_step_num = np.floor(fs_angle / spot_step)
        fs_angle -= fs_angle_step_num * spot_step + spot_step / 2
        events[:, 5] = fs_angle
        return events


def just_decode_daq_data(origin_data, 
                         valid_sum_range: tuple = (100, 32000), 
                         pos_decode_param=200):
    """
    解码DAQ数据
    Args:
        origin_data: 原始事件数据 timestamp + 8道积分值
        channels: 解码使用的通道
        valid_sum_range: 有效事件的积分值范围, 默认为(100, 32000)
        pos_decode_param: 解码参数，默认200
    Returns:
        解码后的有效事件数据
    """
    event_ts = origin_data[:, 0]
    integral_data = origin_data[:, 1:]
    slit_channels = [[4, 5, 6, 7], [0, 1, 2, 3]]
    results = []
    for channels in slit_channels:
        data = integral_data[:, channels]
        data_col_sum = np.sum(data, axis=1)
        valid_mask = (data_col_sum > valid_sum_range[0]) & (data_col_sum < valid_sum_range[1])
        valid_data = data[valid_mask, :]
        valid_event_ts = event_ts[valid_mask]
        valid_energy_data = data_col_sum[valid_mask]
        x = ((valid_data[:, 0] + valid_data[:, 1] - valid_data[:, 2] - valid_data[:, 3]) / valid_energy_data * pos_decode_param) + 255
        y = ((valid_data[:, 0] - valid_data[:, 1] - valid_data[:, 2] + valid_data[:, 3]) / valid_energy_data * pos_decode_param) + 255
        valid_event_mask = (x >= 0) & (y >= 0) & (x < 512) & (y < 512)
        arr = np.column_stack((valid_event_ts[valid_event_mask], x[valid_event_mask], y[valid_event_mask]))
        results.append(arr)
    return results
class TimeClass:
    def __init__(self,
                time_width,  # 0.1ms
                time_step,     # daq采样频率8ns
                angle,
                angle_v  # 角速度1200deg/s
    ):
        self.time_width = time_width
        self.time_step = time_step
        self.bin_width = int(time_width / time_step)
        self.angle_v = angle_v
        self.T_pre = (angle / angle_v) * 1e9

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
                        angle_v=None,
                        time_width=None, time_step=None, T_pre=None,
                        W=5, k=3,
                        show=False):
        if angle_v == None:
            angle_v = self.angle_v
        # if angle_a == None:
        #     angle_a = self.angle_a
        if time_width == None:
            time_width = self.time_width
        if time_step == None:
            time_step = self.time_step
        if T_pre == None:
            T_pre = self.T_pre
            
        # acceleration_time = angle_v / angle_a * 1e9
        T = 18 / angle_v * 1e9
        # start_idx = (acceleration_time + 1e9) / time_width
        # end_idx = cnts.shape[0] -  (acceleration_time / time_width)
        
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
        
        # 上升、下降沿配对
        idx_T_pre = T_pre / time_width
        segs = self.pair_edges_np(rising_idx, falling_idx,
                            T=idx_T_pre,
                            drf_min=0.5 * idx_T_pre,
                            drf_max=1.2 * idx_T_pre,
                            )
        
        segs = segs.astype(int)
        
        # 可视化直接返回当前segs
        if show:
            return segs
        
        for i in range(segs.shape[0]):
            s, e = segs[i, 0], segs[i, 1]
            segs[i, 0] = edges[s]
            segs[i, 1] = edges[e+1]

        T = np.mean(segs[:, 1] - segs[:, 0]) * time_step  # 单位是ns
        self.angle = (T / 1e9) * angle_v
        self.half_angle = self.angle / 2  # 实际扫描角度的一半
        
        return segs

    def pair_edges_np(
        self,
        rising_idx: np.ndarray,
        falling_idx: np.ndarray,
        T: float,
        drf_min: float,
        drf_max: float,
    ) -> np.ndarray:
        """
        输入:
            rising_idx, falling_idx: 升序 np.array
            T: 周期
            drf_min, drf_max: 同一周期内 falling - rising 的允许范围
            period_factor_max: 周期上限系数, 默认 1.2 -> [T, 1.2T]

        输出:
            shape=(N, 2) 的 np.array, 每行 [r, f]
        """
        rising_idx = np.asarray(rising_idx)
        falling_idx = np.asarray(falling_idx)

        i, j = 0, 0
        cand_r = []
        cand_f = []

        # -------- 第一步：按上升沿找对应下降沿（脉宽约束） --------
        while i < len(rising_idx) and j < len(falling_idx):
            r = rising_idx[i]

            # 跳过所有 <= r 的下降沿
            while j < len(falling_idx) and falling_idx[j] <= r:
                j += 1

            if j >= len(falling_idx):
                break

            # 在 [drf_min, drf_max] 范围内找候选下降沿
            # 先看 f - r <= drf_max 能覆盖的索引范围
            # 用 searchsorted 找出上界
            max_f = r + drf_max
            jj_end = np.min([j+3, falling_idx.shape[0]])

            # 当前可行窗口
            window = falling_idx[j:jj_end]
            if window.size > 0:
                dt = window - r
                mask = (dt >= drf_min) & (dt <= drf_max)
                if np.any(mask):
                    # 选第一个满足条件的（最靠近 rising 的）
                    rel_idx = np.argmax(mask)  # 第一个 True 的位置
                    f = window[rel_idx]
                    cand_r.append(r)
                    cand_f.append(f)
                    # j 跳到该 f 后面
                    j = j + rel_idx + 1

            i += 1

        if len(cand_r) == 0:
            return np.empty((0, 2), dtype=rising_idx.dtype)

        candidates = np.column_stack([cand_r, cand_f])
        return candidates

        # -------- 第二步：用周期间隔约束过滤 --------
        # filtered = [candidates[0]]
        # for k in range(1, len(candidates)):
        #     r_prev = filtered[-1][0]
        #     r_curr, f_curr = candidates[k]
        #     dt_r = r_curr - r_prev
        #     if drf_min <= dt_r <= drf_max:
        #         filtered.append([r_curr, f_curr])
        #     else:
        #         # 间隔不在 [T, 1.2T] 内，当成噪声周期丢掉
        #         continue

        # return np.asarray(filtered)



    def segment_by_angle(self, data, segs, angle_v=None, time_step=None):
        """
        将每个周期内的事件按角度划分。

        Args:
            data (np.array): 完整的事件数据 (N, 5)，包含 [timestamp, x, y, rail_x, rail_y]。
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
            delta_T = end_ts - start_ts

            if period_data.shape[0] == 0:
                continue # 如果这个周期没有事件，则跳过

            # 2. 计算每个事件相对于周期开始的时间差（纳秒）
            delta_t = period_data[:, 0] - start_ts

            # 3. 计算每个事件对应的角度
            angles = self.half_angle - (delta_t / delta_T) * self.angle
            # angles = time_elapsed * angle_v_ns - self.half_angle
            
            # 4. 将角度信息和周期ID添加到数据中
            # 创建一个 DataFrame 以方便操作
            df = pd.DataFrame(period_data, columns=['timestamp', 'x', 'y', 'rail_x', 'rail_y'])
            # df['period_id'] = i
            df['angle'] = angles
            
            all_period_events.append(df)
            
            if (i + 1) % 100 == 0: # 每处理100个周期打印一次进度
                print(f"  已处理 {i+1}/{len(segs)} 个周期...")

        if not all_period_events:
            print("警告：在所有识别出的周期内均未找到任何事件。")
            return pd.DataFrame(columns=['timestamp', 'x', 'y', 'energy', 'period_id', 'angle'])

        # 将所有周期的 DataFrame 合并为一个
        final_df = pd.concat(all_period_events, ignore_index=True)
        
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
            output_npy_path = filepath
            print(f"正在保存结果到 {output_npy_path}...")
            # segmented_df.to_csv(output_csv_path, index=False)
            np.save(output_npy_path, segmented_df.to_numpy())
            print("保存完成。")

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
        ax.plot(t_local, xs)  # 纯散点
        
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

    def DetResponse(self, 
                    original_data,
                    angle_bins,
                    img_shape,
                    save=False,
                    save_path=None):
        '''
        根据位置返回分组探测器响应
        data cols: ['timestamp', 'x', 'y', 'energy', 'period_id', 'angle', 'angle_degree', 'rail_x', 'rail_y']
        '''
        # timestamps = original_data.icol[:, 0]
        # edges, counts = self.Time_Counts(timestamps)
        # segs = self.TimeClassification(edges, counts)
        # if segs.size > 0:
        #     segmented_df = self.segment_by_angle(data, segs)
        # else:
        #     print("未找到任何周期，无法进行角度划分。")  
        #     return  
        selected_cols = ['x', 'y', 'period_id', 'angle_degree', 'rail_x', 'rail_y']
        # df = segmented_df[selected_cols]
        df = original_data[selected_cols]
        
        # rail_x 唯一值（保持出现顺序）
        rail_vals = pd.unique(df['rail_x'].astype(int))
        m = len(rail_vals)
        # 预分配结果数组
        imgs = np.zeros((m, angle_bins, img_shape[0], img_shape[1]), dtype=np.int64)
        # 建立 rail_x -> 索引映射
        rail_to_row = {rv: i for i, rv in enumerate(rail_vals)}
        
        # 按 rail_x 分组（不排序，保持天然顺序）
        for rail_val, df_r in df.groupby('rail_x', sort=False):
            i = rail_to_row[int(rail_val)]
            print(i)
            # 按 angle_degree 分组
            for angle_idx, df_ra in df_r.groupby('angle_degree', sort=False):
                pos = df_ra[['x', 'y']].to_numpy()
                if pos.size == 0:
                    continue

                img = self.bins_count_image_from_yx(pos, W=img_shape[0], H=img_shape[1])
                if not isinstance(img, np.ndarray):
                    raise ValueError(
                        f"bins_count_image_from_xy 返回形状 {getattr(img, 'shape', None)}，应为 {img_shape}"
                    )

                imgs[i, int(angle_idx), :, :] = img
        if save:
            np.save(save_path, imgs)

    def bins_count_image_from_yx(self, pos, W=512, H=512, round_mode="truncate"):
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
        return img[::-1, ::-1]

if __name__ == "__main__":
    dir = "./TrueData/test/2025-11-17_15-33_50eda15f"  # 间隔模体数据
    # dir = "./TrueData/test/2025-11-17_15-37_794fc1c8"
    motion_file = dir + "/data/motion.h5"
    daq_origin_data_file = dir + "/data/original_data.h5"
    outputpath0 = dir + "/0_events_with_angle.npy"
    outputpath1 = dir + "/1_events_with_angle.npy"
    print(f"从 {motion_file} 加载导轨数据...")
    print(f"从 {daq_origin_data_file} 加载daq数据...")

    with h5py.File(motion_file, "r") as f:
        rail_motions = f["rail"][:]
        rail_timestamp_offset = f.attrs["rail_timestamp_offset"]
        fs_motions = f["fs"][:]
        fs_timestamp_offset = f.attrs["fs_timestamp_offset"]
    with h5py.File(daq_origin_data_file, "r") as f:
        daq0 = f["daq0"]
        event_ts = daq0["timestamps"][:]
        integral_data = daq0["integral"][:]
        event_data = np.hstack([event_ts.reshape(-1, 1), integral_data])
        start_collecting_time = daq0.attrs["start_collecting_time"]
    
    parser = MotionParser(
            rail_acc_dec=2000, rail_long_axis_speed=5,
            rail_long_axis_min=-82, rail_long_axis_max=28,
            rail_uniform_margin=0.5,
            rail_long_axis="x",
            rail_timestamp_offset=rail_timestamp_offset,
            fs_timestamp_offset=fs_timestamp_offset,
            daq_start_collecting_time=start_collecting_time
        )

    rail_motions = parser.correct_rail_motions(rail_motions)
    rail_motions = parser.refine_rail_motions(rail_motions)
    rail_motions = np.array(rail_motions)
    rail_motions[:, 0] *= 1e9 / 8
    fs_motions = parser.correct_fs_motions(fs_motions, offset=0)
    fs_motions[:, 0] *= 1e9 / 8
    #-----------------此处修改角速度筛选匀速转动部分-----------------#
    fs_angle_v = 1200
    fsmask = fs_motions[:, -1] >= (fs_angle_v - 2)
    fs_t_min = fs_motions[:, 0][fsmask][0]
    # fs_t_min = None
    #-----------------此处修改角速度筛选匀速转动部分-----------------#

    event_data0, event_data1 = just_decode_daq_data(event_data,
                                                    valid_sum_range=(100, 32000), 
                                                    pos_decode_param=200)
    
    print(event_data0[-1, 0])
    
    
    event_with_coords0 = parser.get_count_events_with_coords(
        event_data0, rail_motions, fs_motions, fs_t_min
    )
    event_with_coords1 = parser.get_count_events_with_coords(
        event_data1, rail_motions, fs_motions, fs_t_min
    )
    print(event_with_coords0[-1, 0])
    # event_with_coords = parser.process_fs_angle(event_with_coords, 
    #                                             spot_step=18)
    
    # 特定情况需要根据x位置进行筛选
    # rail_x_mask = event_with_coords[:, 3] > -140
    # data = event_with_coords[:, :-1][rail_x_mask]
    event_with_coords1[:, 2] = 511 - event_with_coords1[:, 2]  #双缝需要镜像

    timeclass = TimeClass(time_width=1e5,  #0.1ms
                          time_step=8,  #8ns
                          angle=18,
                          angle_v=1200  # 1200deg/s
                          )
    
    # timestamps = data[:, 0]
    # edges, counts = timeclass.Time_Counts(timestamps)
    # print(counts.shape)
    # a, b = 0, 3000
    # a, b = 159000, 161399
    # segs = timeclass.TimeClassification(edges, counts, show=True)
    # timeclass.plot_slice_with_segments(counts, segs, a, b)
    
    
    timeclass.SaveCSV(event_with_coords0, outputpath0)
    timeclass.SaveCSV(event_with_coords1, outputpath1)
    
    # np.save(outputpath, event_with_coords)
    