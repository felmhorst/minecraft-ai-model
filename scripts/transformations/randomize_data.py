import random
import numpy as np
from scripts.generation.generate_cuboid import generate_random_cuboid, generate_cuboid
from scripts.generation.generate_pyramid import generate_random_pyramid, generate_pyramid
from scripts.generation.generate_sphere import generate_random_sphere, generate_sphere
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
    inputs = []
    outputs = []
    for i in range(size):
        shapes = {
            "cuboid": generate_random_cuboid,
            "pyramid": generate_random_pyramid,
        }
        shape = random.choice(["cuboid", "pyramid"])

        generate_shape = shapes[shape]
        # hollow = random.choice([True, False])
        hollow = False

        int_array = generate_shape()
        float_array = int_array.astype(np.float32)

        label = f"{'hollow' if hollow else 'solid'} {shape}"
        inputs.append(label)
        outputs.append(float_array)
    return inputs, outputs
