import matplotlib.pyplot as plt
import numpy as np


def visualize_voxel_grid(voxel_grid: np.ndarray):
    occupancy_grid = (voxel_grid > 0)
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(occupancy_grid, edgecolor='k')
    plt.show()
