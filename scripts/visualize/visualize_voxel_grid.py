import matplotlib.pyplot as plt
import numpy as np
import json
from config.paths import BLOCK_MAPPING_ID_TO_TYPE

FALLBACK_COLOR = "#000000"


def visualize_voxel_grid(
        voxel_grids: list[np.ndarray],
        labels: list[str],
        cols: int = 3
):
    """Visualizes multiple voxel grids in a grid view"""

    n = len(voxel_grids)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    # load palette
    with open(BLOCK_MAPPING_ID_TO_TYPE, 'r') as file:
        global_palette_reverse = json.load(file)

    for i, voxel_grid in enumerate(voxel_grids):
        occupancy_grid = (voxel_grid > 0)
        unique_voxels = np.unique(voxel_grid[voxel_grid != 0])

        # set colors
        colors = np.empty(voxel_grid.shape, dtype=object)
        for block_id in unique_voxels:
            filtered_grid = (voxel_grid == block_id)
            block = global_palette_reverse.get(str(block_id), None)
            colors[filtered_grid] = block["color"] if block else FALLBACK_COLOR

        # swap axes for correct display
        occupancy_grid = np.swapaxes(occupancy_grid, 0, 2)
        colors = np.swapaxes(colors, 0, 2)

        # draw subplot
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.voxels(occupancy_grid, facecolors=colors, edgecolor='k')

        # set axis scales
        dx, dy, dz = occupancy_grid.shape
        ax.set_box_aspect([dx, dy, dz])
        ax.set_xlim([0, dx])
        ax.set_ylim([0, dy])
        ax.set_zlim([0, dz])

        # turn of axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # draw title
        ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()
