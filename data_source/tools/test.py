# ../build/spectrum.txt

import numpy as np
import matplotlib.pyplot as plt

# 读取的文件名以及存储的文件名
# 角度字符串，例如 "0", "7p5", "15", "22p5", "30", "37p5", "45"
# 是否有缺陷 "defect" 或 "no_defect"
# 是否为单次散射 "single" 或无

angle_str = "0"
defect_str = "interval"  # "defect" 表示有缺陷，"no_defect" 表示无缺陷，"interval" 表示间隔缺陷
single_str = "single"  # "single" 表示单次散射，仅影响保存的文件名
interval_mm = 3  # 间隔缺陷的间隔，单位毫米，仅在 defect_str 为 "interval" 时使用


# /home/erinneria/data_source/output/spectrum.txt
# data = np.loadtxt('/home/erinneria/data_source/output/spectrum_0_n.txt')
# data = np.loadtxt(f'/home/erinneria/data_source/output/spectrum_{angle_str}_{"d" if defect_str=="defect" else ("interval" if defect_str=="interval" else "n")}.txt')
data = np.loadtxt(f'/home/erinneria/data_source/output/spectrum_{angle_str}_{"defect" if defect_str=="defect" else ("interval_"+str(interval_mm)+"mm" if defect_str=="interval" else "no_defect")}.txt')

# 第二列是检测到该粒子的探测器单元，探测器是50乘以50的阵列
# 探测器单元计数二维热图
detector_size = 50
spectrum = np.zeros((detector_size, detector_size))
#第五列是散射次数，我只要散射一次的粒子
# data = data[data[:, 4] == 1]
if single_str == "single":
    data = data[data[:, 4] == 1]
    
for det_unit in data[:, 1]:
    if det_unit < 0:
        continue  # 忽略未被探测到的粒子
    x = int(det_unit) // detector_size
    y = int(det_unit) % detector_size
    spectrum[x, y] += 1
# np.save('/home/erinneria/data_source/tools/0_degree_no_defect_single.npy', spectrum)
# np.save(f'/home/erinneria/data_source/tools/{angle_str}_degree_{"defect" if defect_str=="defect" else "no_defect"}{"_single" if single_str=="single" else ""}.npy', spectrum)
# np.save(f'/home/erinneria/data_source/tools/{angle_str}_degree_{"defect" if defect_str=="defect" else ("interval" if defect_str=="interval" else "no_defect") }{"_single" if single_str=="single" else ""}.npy', spectrum)
np.save(f'/home/erinneria/data_source/tools/{angle_str}_degree_{"defect" if defect_str=="defect" else ("interval_"+str(interval_mm)+"mm" if defect_str=="interval" else "no_defect") }{"_single" if single_str=="single" else ""}.npy', spectrum)

# 可视化某个入射角度下的探测器单元计数热图
plt.figure(figsize=(8, 6))
plt.imshow(spectrum, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title(f'Detector Counts at Angle {angle_str} deg')
plt.xlabel('Detector X Unit')
plt.ylabel('Detector Y Unit')

plt.figure(figsize=(8, 6))
plt.plot(spectrum.sum(axis=0))
plt.xlabel('Detector X Unit')
plt.ylabel('Counts')
plt.title('Sum of Detector Counts Across Y Units')
plt.show()
