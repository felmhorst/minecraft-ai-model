import numpy as np


def generate_sphere(center=(8, 8, 8), radius=4, hollow=False, grid_size=16):
    # calculate distance to center per voxel
    x, y, z = np.indices((grid_size, grid_size, grid_size))
    dist_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2

    # generate grid & fill with voxels
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    if hollow:
        # Voxels close to the surface (radius^2 - 1 <= d^2 <= radius^2)
        shell_thickness = 1  # Thickness of the hollow shell
        grid[(dist_squared >= (radius - shell_thickness) ** 2) & (dist_squared <= radius ** 2)] = 1
    else:
        grid[dist_squared <= radius ** 2] = 1
    return grid


def generate_random_sphere(hollow=False, grid_size=16):
    """generates a sphere of random size"""

    # randomize size
    radius = np.random.randint(2, (grid_size // 2) - 1)

    # randomize position
    center = np.array([
        np.random.randint(radius, grid_size - radius),
        np.random.randint(radius, grid_size - radius),
        np.random.randint(radius, grid_size - radius)
    ])

    return generate_sphere(center, radius, hollow, grid_size)
