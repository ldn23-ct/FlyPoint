import numpy as np

def Mapping(pos: np.ndarray, voxelsize: np.ndarray, kernalsize: np.ndarray):
    '''
    input
      pos--模体前表面左上角点坐标 初始时刻对准零点 ndarray of shape (2,) dtype=float
      dx/dy/dz--空间体素大小 根据空间坐标计算体素编号
      kernal_y/kernal_z--二维卷积核尺寸 大小与体素大小相同 计算会极大简化
    output
      map--二维卷积核每个像素与三维空间体素的对应关系
    idea
      (1) 空间体素划分编号 n = iz + iy * nz + ix * (ny + nz)
      (2) 根据 pos_x 找到 ix
      (3) 假设零点是对准的 kernal相当于三维矩阵的一个切片 只需要根据 pos_y 就可以求出 iy0
          并顺便取出每一行对应的 iy 
      (4) 当 pos_x 位于交界线上 归 x- 像素
      (5) 当 iy 越界 map赋值null
    '''
    ix = -pos[0] // voxelsize[0]

    return

def DetArray(corners: list, pixelsize: list, detsize: list):
    '''
    input
      corner_pos--4 corners position of the detector
      dx/dy--length of the detector pixel
      x/y--length of the detector
    output
      detarray--pos of the pixels' center shape:[ny, nx]
    idea
      (1) 计算像素个数
      (2) 计算每个像素中心点
    '''
    nx, ny = detsize[0] / pixelsize[0], detsize[1] / pixelsize[1]
    P0, P1, P2, P3 = map(np.array, corners)
    centers = []

    for i in range(ny):
        v_top_start = P0 + (P3 - P0) * i / ny
        v_top_end   = P1 + (P2 - P1) * i / ny
        v_bot_start = P0 + (P3 - P0) * (i + 1) / ny
        v_bot_end   = P1 + (P2 - P1) * (i + 1) / ny

        for j in range(nx):
            # Compute 4 corners of the sub-rectangle
            tl = v_top_start + (v_top_end - v_top_start) * j / nx        # top-left
            # tr = v_top_start + (v_top_end - v_top_start) * (j + 1) / nx  # top-right
            br = v_bot_start + (v_bot_end - v_bot_start) * (j + 1) / nx  # bottom-right
            # bl = v_bot_start + (v_bot_end - v_bot_start) * j / nx        # bottom-left

            center = (tl + br) / 2  # diagonal midpoint
            centers.append(center)

    centers = np.array(centers).reshape(ny, nx)
    return centers

def ObjArray(angle, fdis, bdis, kernalsize):
  '''
  input
    angle--the angle of the fan beam
    fdis--the distance from source to the front surface of the object
    bdis--the distance from source to the back surface of the object
    kernel_y/kernel_z--size of kernal, equal to voxel
  output
    objarray--pos of the voxels' center
  '''
  angle = np.deg2rad(angle)
  height = bdis * np.tan(angle/2)
  ny = np.ceil(height / kernalsize[0])
  nz = np.ceil((bdis - fdis) / kernalsize[1])
  centers = np.zeros((ny*nz, 2))
  y0, z0 = height - kernalsize[0]/2, fdis + kernalsize[1]/2
  for i in range(ny):
      for j in range(nz):
          idx = j + i * nz
          centers[idx] = [y0 - i * kernalsize[0], z0 + j * kernalsize[1]]
  return centers

