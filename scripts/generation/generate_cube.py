import numpy as np
import random


def generate_cube():
    grid = np.zeros((16, 16, 16), dtype=int)

    # Define cube size (minimum 3x3x3 to be hollow)
    size_x = random.randint(6, 14)
    size_z = random.randint(6, 14)
    height = random.randint(6, 14)

    # Random position ensuring it fits in the grid
    max_x = 16 - size_x
    max_z = 16 - size_z
    start_x = random.randint(0, max_x)
    start_z = random.randint(0, max_z)
    start_y = 0
    end_x = start_x + size_x
    end_z = start_z + size_z
    end_y = start_y + height

    if end_y > 16:
        height = 16 - start_y
        end_y = start_y + height

    # Fill only the outer shell (hollow cube)
    for y in range(start_y, end_y):
        for z in range(start_z, end_z):
            for x in range(start_x, end_x):
                is_edge = (
                        x == start_x or x == end_x - 1 or
                        z == start_z or z == end_z - 1 or
                        y == start_y or y == end_y - 1
                )
                if is_edge:
                    grid[y, z, x] = 1

    return grid
