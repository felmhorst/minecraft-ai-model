from typing import Tuple

import numpy as np
from numpy import ndarray

from scripts.generation.get_random_material_with_label import get_random_material_with_label


def generate_sphere(
    center: tuple[int, int, int] = (8, 8, 8),
    radius: int = 4,
    is_hollow: bool = False,
    grid_size: int = 16,
    solid_block_id: int = 1,
) -> np.ndarray:
    """generates a sphere of the specified dimensions on a voxel grid"""
    # calculate distance to center per voxel
    x, y, z = np.indices((grid_size, grid_size, grid_size))
    dist_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2

    # generate grid & fill with voxels
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    if is_hollow:
        # Voxels close to the surface (radius^2 - 1 <= d^2 <= radius^2)
        wall_thickness = 1
        grid[(dist_squared >= (radius - wall_thickness) ** 2) & (dist_squared <= radius ** 2)] = solid_block_id
    else:
        grid[dist_squared <= radius ** 2] = solid_block_id
    return grid


def generate_random_sphere(
    is_hollow: bool = False,
    grid_size: int = 16
) -> tuple[ndarray, str]:
    """generates a sphere of random size with a label"""

    # randomize size
    radius = np.random.randint(2, (grid_size // 2) - 1)

    # randomize position
    center = (
        np.random.randint(radius, grid_size - radius),
        np.random.randint(radius, grid_size - radius),
        np.random.randint(radius, grid_size - radius)
    )

    # randomize material
    material_id, material_label = get_random_material_with_label()

    # label
    shape_label = np.random.choice(["sphere", "ball", "globe"], p=[.5, .3, .2])
    hollowness_label = np.random.choice(["hollow " if is_hollow else "solid ", ""])
    label = f"{hollowness_label}{material_label}{shape_label}"

    return generate_sphere(center, radius, is_hollow, grid_size, material_id), label
