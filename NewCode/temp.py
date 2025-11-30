import numpy as np
import matplotlib.pyplot as plt
from geo_proj import CalSysTool
from scipy.io import loadmat

def slit_simulation():
    slit = np.array([[-27.975, -25, -36.97], [-27.975, 25, -36.97]])
    src = np.array([0, 0, 158])

    toolbox = CalSysTool(src_pos=src,
                         slit=slit,
                         calibrationdata_path=None,
                         )
    # 单缝标定模型
    cols_calibration = np.load("./data/Calibration_data/cols_calibration0.npy")
    ds_calibration = np.arange(0, 62, 2)
    prefunc0, model0 = toolbox.PreCols(ds_calibration, cols_calibration)
    
    # 生成方波信号
    A, B = model0
    ds = np.arange(0, 60, 0.1)
    dcs = B*ds / (1-A*ds)
    cs = cols_calibration[0] + dcs
    mu = -0.1276 * 2.699 / 10
    value = ((ds // 10) % 2 == 0).astype(int)
    delta_d = (np.full_like(ds[1:], ds[1]-ds[0]))*value[1:]
    for i in range(1, delta_d.shape[0]):
        delta_d[i] = delta_d[i] + delta_d[i-1]
    amplitude = np.ones_like(ds)    
    amplitude[1:] = np.exp(mu * delta_d)
    single = amplitude * value
    new_col = np.arange(np.floor(cols_calibration[0]), np.ceil(cs[-1]))
    new_single = np.interp(new_col, cs, single)
    
    # 定义“偏深侧”的高斯核 h(z)
    dcol = 1
    sigma = 3.0 / (48 / 512)  # 核宽度 3mm
    delta = 2.0 / (48 / 512)  # 核中心向深侧偏移 2mm

    # 核的 z 轴：取一个足够大的对称范围，例如 ±5σ
    kernel_half_width = 5 * sigma
    z_kernel = np.arange(-kernel_half_width, kernel_half_width + dcol, dcol)

    # 偏移高斯
    h = np.exp(-(z_kernel - delta)**2 / (2 * sigma**2))
    h /= h.sum()  # 归一化，使 sum(h) = 1
    
    # 卷积
    g = np.convolve(new_single, h, mode='same')
    
    plt.plot(new_col, new_single, label="Original")
    plt.plot(new_col, g, label="Convolved")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    data_dir = "./data/Calibration_data"
    labelmap_list = []
    for i in range(2):
        mappath = data_dir + f"/labelmap{i}.mat"
        labelmap_list.append(loadmat(mappath)["labelmap"])
    cols0 = np.loadtxt("./data/Calibration_data/cols_calibration0.txt", delimiter=" ", encoding="utf-8")
    cols1 = np.loadtxt("./data/Calibration_data/cols_calibration1.txt", delimiter=" ", encoding="utf-8")
    cols = np.column_stack((cols0, cols1))
    