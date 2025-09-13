import numpy as np
from numpy import random

from scripts.generation.get_random_material_with_label import get_random_material_with_label


def generate_pyramid(
    position: tuple[int, int, int] = (4, 4, 4),
    base_width: int = 7,
    is_hollow: bool = False,
    grid_size: int = 16,
    solid_block_id: int = 1,
) -> np.ndarray:
    """generates a pyramid of the specified dimensions on a voxel grid"""
    # calculate height
    height = base_width // 2 + (1 if base_width % 2 != 0 else 0)
    start_x = position[0]
    start_y = position[1]
    start_z = position[2]

    # generate grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    for level in range(height):
        y = start_y + level
        layer_width = base_width - 2 * level
        if layer_width <= 0 or y >= grid_size:
            break

        x1 = start_x + level
        z1 = start_z + level
        x2 = x1 + layer_width
        z2 = z1 + layer_width

        for z in range(z1, z2):
            for x in range(x1, x2):
                if not is_hollow:
                    grid[y, z, x] = solid_block_id
                else:
                    d_left = x - x1
                    d_right = (x2 - 1) - x
                    d_front = z - z1
                    d_back = (z2 - 1) - z
                    wall_thickness = 2
                    # draw walls that are ≤ wall_thickness‑1 blocks thick
                    if (min(d_left, d_right, d_front, d_back) < wall_thickness
                            or layer_width <= wall_thickness * 2 - 1):
                        grid[y, z, x] = solid_block_id

    return grid


def generate_random_pyramid(
    is_hollow: bool = False,
    grid_size: int = 16
) -> tuple[np.ndarray, str]:
    """generates a pyramid of random size with a label"""

    # randomize size
    base_width = np.random.randint(5, grid_size)

    # randomize position
    max_x = grid_size - base_width
    max_z = grid_size - base_width
    start_x = np.random.randint(0, max_x)
    start_z = np.random.randint(0, max_z)
    start_y = 0
    position = (start_x, start_y, start_z)

    # randomize material
    material_id, material_label = get_random_material_with_label()

    # label
    size_label = np.random.choice(["big " if base_width > 8 else "small ", ""])
    hollowness_label = np.random.choice(["hollow " if is_hollow else "solid ", ""])
    label = f"{size_label}{hollowness_label}{material_label}pyramid"

    return generate_pyramid(position, base_width, is_hollow, grid_size, material_id), label
