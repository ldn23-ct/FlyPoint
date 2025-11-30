from base import BASE_DIR

import argparse
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def cal_scatter_graph(data, valid_sum_range: tuple = (100, 32000), decode_pos_params: int = 200):
    data_col_sum = np.sum(data, axis=1)
    valid_mask = (data_col_sum > valid_sum_range[0]) & (data_col_sum < valid_sum_range[1])
    valid_data = data[valid_mask, :]
    valid_energy_data = data_col_sum[valid_mask]
    x = np.round((valid_data[:, 0] + valid_data[:, 1] - valid_data[:, 2] - valid_data[:, 3]) / valid_energy_data * decode_pos_params) + 255
    y = np.round((valid_data[:, 0] - valid_data[:, 1] - valid_data[:, 2] + valid_data[:, 3]) / valid_energy_data * decode_pos_params) + 255
    valid_event_mask = (x >= 0) & (y >= 0) & (x < 512) & (y < 512)
    valid_x = x[valid_event_mask].astype(int)
    valid_y = y[valid_event_mask].astype(int)
    
    img = np.zeros((512, 512), dtype=np.int32)
    np.add.at(img, (valid_x, valid_y), 1)
    return img[:, ::-1]


def read_integral_data(origin_file, channels: list, daq_name: str = "daq0"):
    if len(channels) != 4:
        return None
    origin_file = Path(origin_file)
    if not origin_file.exists():
        raise FileNotFoundError(f"File {origin_file} not found")
    if origin_file.is_file():
        # 是文件
        if origin_file.suffix == ".h5":
            with h5py.File(origin_file, "r") as f:
                if daq_name not in f:
                    return None
                else:
                    daq = f[daq_name]
                    integral = daq["integral"][:]
        elif origin_file.suffix == ".txt":
            df = pd.read_csv(origin_file, sep=" ", header=None)
            data = df.to_numpy()
            if data.shape[1] == 10:
                if "0" in daq_name:
                    mask = data[:, 0] == 0
                elif "1" in daq_name:
                    mask = data[:, 0] == 1
                else:
                    return None
                integral = data[mask, 2:]
            elif data.shape[1] == 9:
                integral = data[:, 1:]
            else:
                return None
        else:
            return None
        return integral[:, channels]
    elif origin_file.is_dir():
        # 是目录
        files = list(origin_file.glob("*.h5"))
        if not files:
            files = list(origin_file.glob("*.txt"))
        if not files:
            return None
        for file in files:
            integral = read_integral_data(file, channels, daq_name)
            if integral is not None:
                return integral
        return None
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--origin_file", type=str, help="原始数据文件或目录")
    parser.add_argument("-c", "--channels", type=int, nargs=4, help="四个通道的索引")
    parser.add_argument("-d", "--daq_name", type=str, default="daq0", help="数据来源的DAQ名称")
    parser.add_argument("-s", "--valid_sum_range", type=int, nargs=2, default=[100, 32000], help="有效数据范围")
    return parser.parse_args()


if __name__ == '__main__':
    # python tools/display_scatter_graph.py -o "D:\Work\背散射项目\2025-11-10日出差\TwinDAQ\2025_11_12_19_40_15" -c 4 5 6 7 -d daq0 -s 100 32000
    args = parse_args()
    origin_file = args.origin_file
    channels = args.channels
    valid_sum_range = args.valid_sum_range
    data = read_integral_data(origin_file, channels, daq_name=args.daq_name)
    if data is None:
        print("No data found")
        exit(1)
    
    img = cal_scatter_graph(data, valid_sum_range=valid_sum_range)
    
    display_img = img.copy()
    high_thr = np.percentile(display_img, 99.9)
    low_thr = np.percentile(display_img, 0.1)
    display_img[display_img > high_thr] = high_thr
    display_img[display_img < low_thr] = low_thr

    plt.imshow(display_img, cmap="gray")
    plt.show()
    
    cols = np.sum(img, axis=0)
    max_val, min_val = np.max(cols), np.min(cols[cols > 0])
    print("max: ", max_val, "min: ", min_val)
    print("decay: ", min_val / max_val)
    plt.plot(cols)
    plt.show()
