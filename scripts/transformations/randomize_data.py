from scripts.transformations.flip import random_flip
from scripts.transformations.rotate import random_rotate
from scripts.transformations.translate import random_safe_translate_x, random_safe_translate_z


def randomize_data(data_3d):
    """applies random valid transformations to the data"""
    return random_safe_translate_x(
        random_safe_translate_z(
            random_rotate(
                random_flip(data_3d)
            )
        )
    )