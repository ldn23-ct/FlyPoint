# ！/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time          : 2025/1/8 下午1:19
# @Author        : OuyangXujian
# @File          : motion_tracker.py
# @Project       : backscatter_system
# @Description   : 运动记录

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import numpy as np
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

if __name__ == "__main__":
    # task_dir = r"./TrueData/taskstasks/2025-10-29_16-26_e3c8fa25"
    # motion_file = Path(task_dir) / "data" / "motion.h5"
    # daq_origin_data_file = Path(task_dir) / "data" / "original_data.h5"
    motion_file = "./TrueData/tasks/2025-10-29_16-26_e3c8fa25/data/motion.h5"
    daq_origin_data_file = "./TrueData/tasks/2025-10-29_16-26_e3c8fa25/data/original_data.h5"
    
    with h5py.File(motion_file, "r") as f:
        motions = f["rail"][:]
        rail_timestamp_offset = f.attrs["rail_timestamp_offset"]
    
    with h5py.File(daq_origin_data_file, "r") as f:
        daq0 = f["daq0"]
        event_ts = daq0["timestamps"][:]
        start_collecting_time = daq0.attrs["start_collecting_time"]
    
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
    motions[:, 0] *= 1e9  # 时间单位转回ns
    event_ts *= 8 # 时间单位从8ns转回ns
    
    valid_event_rail_xy, valid_event_ts, event_mask = parser.get_count_events_with_coords(event_ts, motions)
    print(valid_event_rail_xy.shape)
    print(valid_event_ts.shape)
