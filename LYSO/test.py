import os
import numpy as np
import matplotlib.pyplot as plt
import math
import IntersectionLength as IL

if __name__ == "__main__":
    obj_size = np.array([4, 4, 20])  # mm
    source2object = 10  # mm
    source2point = 25  # mm
    grid_start = np.array([-obj_size[0]/2, -obj_size[1]/2, source2object])
    grid_end = np.array([ obj_size[0]/2, obj_size[1]/2, source2object + obj_size[2]])
    nx,ny,nz = 4,4,100
    dx = (grid_end[0]-grid_start[0])/nx
    dy = (grid_end[1]-grid_start[1])/ny
    dz = (grid_end[2]-grid_start[2])/nz
    # grid_size = np.array([dx,dy,dz])
    voxel_pos = np.zeros((nz, 3))
    for iz in range(nz):
        idx = iz
        y_pos = 0
        z_pos = grid_start[2] + (iz + 0.5) * dz
        voxel_pos[idx,:] = np.array([0, y_pos, z_pos])

    n_det = 48
    cell_x = np.linspace(-detector_size_x/2, detector_size_x/2, n_det)
    det_cell_pos = np.zeros((n_det, n_det, 3))
    for i in range(n_det):
        offset = cell_x[i]
        det_cell_pos[i,:,:] =  det_xvec * offset + detector_pos
    det_cell_pos = np.reshape(det_cell_pos, (n_det * n_det, 3))