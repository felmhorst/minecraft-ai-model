import random

import numpy as np

from scripts.generation.generate_cube import generate_cube
from scripts.training_data import load_training_data
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


def get_random_data():
    input_texts, outputs = load_training_data()
    i = random.randint(0, len(input_texts) - 1)
    return input_texts[i], randomize_data(np.array(outputs[i]))


def get_random_dataset(size):
    outputs = []
    for i in range(size):
        int_array = generate_cube()
        float_array = int_array.astype(np.float32)
        outputs.append(float_array)
    return outputs
