from scripts.generation.generate_cuboid import generate_random_cuboid
from scripts.generation.generate_pyramid import generate_random_pyramid
from scripts.generation.generate_sphere import generate_random_sphere
from scripts.training_data.get_random_training_sample import get_random_training_sample
from numpy.random import choice


def get_random_training_dataset(size: int) -> tuple[list, list]:
    inputs = []
    outputs = []

    generate_shape = {
        "cuboid": generate_random_cuboid,
        "pyramid": generate_random_pyramid,
        "sphere": generate_random_sphere,
    }

    for i in range(size):
        draw = choice(["cuboid", "pyramid", "sphere", "other"], 1, p=[.2, .2, .2, .4])[0]
        if draw == "other":
            label, data = get_random_training_sample()
        else:
            is_hollow = choice([True, False], 1)[0]
            data = generate_shape[draw](hollow=is_hollow)
            label = f"{'hollow' if is_hollow else 'solid'} {draw}"
        inputs.append(label)
        outputs.append(data)
    return inputs, outputs
