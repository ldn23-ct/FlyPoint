import numpy as np

def Mapping():
    '''
    input
      pos--射线初始时刻对准零点 ndarray of shape (2,) dtype=float
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
    return

def DetArray():
    '''
    input
      corner_pos--4 corners position of the detector
      dx/dy--length of the detector pixel
      x/y--length of the detector
    output
      detarray--pos of the pixels' center
    idea
      (1) 计算像素个数
      (2) 计算每个像素中心点
    '''
    return

def ObjArray():
  '''
  input
    angle--the angle of the fan beam
    fdis--the distance from source to the front surface of the object
    bdis--the distance from source to the back surface of the object
    kernel_y/kernel_z--size of kernal, equal to voxel
  output
    objarray--pos of the voxels' center
    fanmask--0/1 matrix, shape: [p, m*n], voxels which be irradiated
  '''