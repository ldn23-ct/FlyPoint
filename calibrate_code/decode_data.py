# ！/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time          : 2024/10/31 下午12:44
# @Author        : OuyangXujian
# @File          : decode_original_data.py
# @Project       : backscatter_system
# @Description   : 解码原始数据并筛选出有效数据

from base import BASE_DIR

import os
import argparse
import h5py
import numpy as np
import time
from tqdm import tqdm
from control.twindaq.decoder import DetectorDecoder
from control.twindaq.writer import AsyncDataWriter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-file', type=str, required=True,
                        help="Path of original data file")
    parser.add_argument('--output-file', type=str, required=True,
                        help="Path of output file")
    parser.add_argument('--daq-mat-file', type=str,
                        default="./resources/control/daq/GAGG_1_calibration.h5",
                        help="Path of daq calibration matrix file")
    parser.add_argument('--pos-decode-param', type=int, default=200)
    parser.add_argument('--valid-energy-range', type=float, nargs='+', default=[10, 20000],
                        help="the valid energy range")
    parser.add_argument('--batch-size', type=int, default=1e6,
                        help='Set the batch data size to prevent memory overflow due to large data size.')
    return parser.parse_args()


def load_daq_mat(h5_file):
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            classification_map = f.get('classification_map')
            if classification_map is not None:
                classification_map = classification_map[:]
            position_map = f.get('position_map')
            if position_map is not None:
                position_map = position_map[:]
            kandb = f.get('kandb')
            if kandb is not None:
                kandb = kandb[:]
            mat_dict = dict(classification_map=classification_map,
                            position_map=position_map,
                            kandb=kandb,)
            return mat_dict
    else:
        return None


def main(args):
    if len(args.valid_energy_range) != 2:
        return
    if not os.path.exists(args.origin_file):
        return
    daq_mat_dict = load_daq_mat(args.daq_mat_file)
    decoder = DetectorDecoder(daq_id=0, valid_event_range=args.valid_energy_range, 
                              pos_decode_param=args.pos_decode_param, 
                              daq_timestamp_interval=8e-9,
                              **daq_mat_dict)
    decoder.set_start_collecting_time(0.0)
    writer = AsyncDataWriter()
    write_path = args.output_file
    write_dir = os.path.dirname(write_path)
    write_key = os.path.splitext(os.path.basename(write_path))[0]
    writer.set_data_save_path(write_dir)
    with h5py.File(args.origin_file, 'r') as f:
        if "start_collecting_time" in f.attrs.keys():
            start_collecting_time = f.attrs['start_collecting_time']
            decoder.set_start_collecting_time(start_collecting_time)
        daq0 = f["daq0"]
        data_len = daq0["integral"].shape[0]
        single_data_len = daq0["integral"].shape[1]
        writer.set_channels(single_data_len)
        batch_data_size = int(args.batch_size)
        batch_data_len = int(np.ceil(data_len / batch_data_size))
        for i in tqdm(range(batch_data_len)):
            data_start_idx = i * batch_data_size
            data_end_idx = min((i + 1) * batch_data_size, data_len)
            integral_data = daq0["integral"][data_start_idx:data_end_idx]
            timestamps = daq0["timestamps"][data_start_idx:data_end_idx]
            data = np.hstack([timestamps.reshape(-1, 1), integral_data, integral_data])
            valid_timestamp, position, energy = decoder.decode_original_data(data, from_device=False)
            if valid_timestamp is None:
                continue
            writer.append_data_to_file(write_key, "valid_timestamp", valid_timestamp)
            writer.append_data_to_file(write_key, "position", position)
            writer.append_data_to_file(write_key, "energy", energy)

    writer.close_all_handlers()


if __name__ == '__main__':
    # python tools/decode_original_data.py --origin-file "./storage/temp/2024-09-30_14-56-55_50521c3d-612f-453d-a59d-a3667376ac65_daq.h5" --output-file "./storage/temp/test_processed.h5"
    args = parse_arguments()
    start = time.monotonic()
    main(args)
    end = time.monotonic()
    print(f"Finished in {end - start:.2f}s")
