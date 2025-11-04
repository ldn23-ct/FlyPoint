import numpy as np
import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name} shape={obj.shape}, dtype={obj.dtype}")

chunk_size = 1_000_000
output = np.memmap('merged.dat', dtype=np.int32, mode='w+', shape=(33069440, 9))
with h5py.File("./TrueData/tasks/2025-10-29_15-54_f740c5d5/data/original_data.h5", 'r') as f:
    timestamps = f['daq0/timestamps']
    integral = f['daq0/integral']
    peak = f['daq0/peak']
    
    n = timestamps.shape[0]
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        print(f"Processing rows {i}:{end}")

        chunk_ts = timestamps[i:end].reshape(-1, 1)
        chunk_integral = integral[i:end, 0:4]
        chunk_peak = peak[i:end, 0:4]

        output[i:end, :] = np.hstack((chunk_ts, chunk_integral, chunk_peak))

# 保存为 .npy 或 .csv
np.save('merged.npy', output)