import numpy as np
from numpy import random

from scripts.generation.get_random_material_with_label import get_random_material_with_label


def generate_cuboid(
    position: tuple[int, int, int] = (4, 4, 4),
    width: int = 8,
    length: int = 8,
    height: int = 8,
    is_hollow: bool = False,
    grid_size: int = 16,
    solid_block_id: int = 1,
) -> np.ndarray:
    """generates a cuboid of the specified dimensions on a voxel grid"""
    start_x = position[0]
    start_y = position[1]
    start_z = position[2]
    end_x = start_x + width
    end_z = start_z + length
    end_y = start_y + height

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    if end_y > grid_size:
        height = grid_size - start_y
        end_y = start_y + height

    # Fill only the outer shell (hollow cube)
    for y in range(start_y, end_y):
        for z in range(start_z, end_z):
            for x in range(start_x, end_x):
                if is_hollow:
                    is_solid_voxel = (
                        x == start_x or x == end_x - 1 or
                        z == start_z or z == end_z - 1 or
                        y == start_y or y == end_y - 1
                    )
                    if is_solid_voxel:
                        grid[y, z, x] = solid_block_id
                else:
                    grid[y, z, x] = solid_block_id

    return grid


def generate_random_cuboid(
    is_hollow: bool = False,
    grid_size: int = 16
) -> tuple[np.ndarray, str]:
    """generates a cuboid of random size with a label"""

    # randomize size
    width = np.random.randint(5, grid_size)
    length = np.random.randint(5, grid_size)
    height = np.random.randint(5, grid_size)

    # randomize position
    max_x = grid_size - width
    max_z = grid_size - length
    start_x = np.random.randint(0, max_x)
    start_z = np.random.randint(0, max_z)
    start_y = 0
    position = (start_x, start_y, start_z)

    # randomize material
    material_id, material_label = get_random_material_with_label()

    # label
    is_cube = width == length and width == height
    is_tall = height > width and height > length
    shape_label = "cube" if is_cube else "cuboid"
    tall_label = np.random.choice(["tall " if is_tall else "", ""])
    hollowness_label = np.random.choice(["hollow " if is_hollow else "solid ", ""])
    label = f"{tall_label}{hollowness_label}{material_label}{shape_label}"

    return generate_cuboid(position, width, length, height, is_hollow, grid_size, material_id), label


