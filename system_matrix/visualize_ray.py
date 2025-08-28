import matplotlib.pyplot as plt
import numpy as np

def visualize_ray(vector, angle_index):
    """
    在 yz 平面可视化指定角度下的二维矩阵
    vector: (nray, ny, nz, 3)
    angle_index: 选择的射线角度索引
    ny, nz: 网格大小
    """
    # 取出指定角度下的光线矩阵
    slice_matrix = vector[angle_index, :, :, :]
    
    # 判断哪些 voxel 有值（强度非零）
    mask = np.linalg.norm(slice_matrix, axis=-1) > 0  # (ny, nz)，True 表示有光线
    
    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="Reds", origin="lower")
    plt.title(f"Ray Visualization (angle_index={angle_index})")
    plt.xlabel("z index")
    plt.ylabel("y index")
    plt.colorbar(label="Ray Hit (1=True, 0=False)")
    plt.show()
