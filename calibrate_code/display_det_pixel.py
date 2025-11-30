from base import BASE_DIR

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from display_scatter_graph import read_integral_data
from decode_data import load_daq_mat


def cal_det_pixel_image(integral_data, daq_mat_dict, det_size,
                        valid_sum_range: tuple = (100, 32000), decode_pos_params: int = 200):
    classification_map = daq_mat_dict["classification_map"]
    position_map = daq_mat_dict["position_map"]
    
    data_col_sum = np.sum(integral_data, axis=1)
    valid_mask = (data_col_sum > valid_sum_range[0]) & (data_col_sum < valid_sum_range[1])
    valid_data = integral_data[valid_mask, :]
    valid_energy_data = data_col_sum[valid_mask]
    x = np.round((valid_data[:, 0] + valid_data[:, 1] - valid_data[:, 2] - valid_data[:, 3]) / valid_energy_data * decode_pos_params) + 255
    y = np.round((valid_data[:, 0] - valid_data[:, 1] - valid_data[:, 2] + valid_data[:, 3]) / valid_energy_data * decode_pos_params) + 255
    valid_event_mask = (x >= 0) & (y >= 0) & (x < 512) & (y < 512)
    valid_x = x[valid_event_mask].astype(int)
    valid_y = y[valid_event_mask].astype(int)
    idx = classification_map[valid_x, valid_y]
    position = position_map[idx, :]
    print(np.max(position, axis=0), np.min(position, axis=0))
    img = np.zeros(det_size)
    np.add.at(img, (position[:, 0], position[:, 1]), 1)
    return img


def main(args):
    if len(args.valid_sum_range) != 2:
        return
    if not os.path.exists(args.origin_file):
        return
    daq_mat_dict = load_daq_mat(args.daq_mat_file)

    data = read_integral_data(args.origin_file, 
                              channels=args.channels,
                              daq_name=args.daq_name)
    det_pix_img = cal_det_pixel_image(data, daq_mat_dict, det_size=args.det_size)
    
    plt.imshow(det_pix_img, cmap="gray")
    plt.show()
    
    plt.plot(np.sum(det_pix_img, axis=0))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--origin_file", type=str, default=r"C:\Users\oyxjj\Downloads\2025_11_20_10_38_7", help="原始数据文件或目录")
    parser.add_argument("-c", "--channels", type=int, nargs=4, default=[0, 1, 2, 3], help="四个通道的索引")
    parser.add_argument("-n", "--daq_name", type=str, default="daq0", help="数据来源的DAQ名称")
    parser.add_argument("-s", "--valid_sum_range", type=int, nargs=2, default=[100, 32000], help="有效数据范围")
    parser.add_argument("-d", "--det_size", type=int, nargs=2, default=[44, 44], help="探测器尺寸")
    parser.add_argument("-m", "--daq_mat_file", type=str, default=os.path.join(BASE_DIR, "resources/control/daq/GAGG_1_calibration.h5"), help="数据来源的DAQ校准文件")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
