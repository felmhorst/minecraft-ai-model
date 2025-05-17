import numpy as np


def generate_cuboid(position=(4, 4, 4), width=8, length=8, height=8, hollow=False, grid_size=16):
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
                if hollow:
                    is_solid = (
                            x == start_x or x == end_x - 1 or
                            z == start_z or z == end_z - 1 or
                            y == start_y or y == end_y - 1
                    )
                    if is_solid:
                        grid[y, z, x] = 1
                else:
                    grid[y, z, x] = 1

    return grid


def generate_random_cuboid(hollow=False, grid_size=16):
    """generates a cuboid of random size"""

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
    position = [start_x, start_y, start_z]

    return generate_cuboid(position, width, length, height, hollow, grid_size)
