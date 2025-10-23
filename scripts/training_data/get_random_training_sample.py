import numpy as np
from scripts.training_data.prepare_training_data import load_training_data
from scripts.transformations.randomize_data import randomize_data


def get_random_training_sample() -> tuple[np.ndarray, str]:
    data_groups = load_training_data()

    # random group
    group_index = np.random.randint(0, len(data_groups))
    group = data_groups[group_index]

    # random label
    label_index = np.random.randint(0, len(group['labels']))
    label = group['labels'][label_index]

    # random voxel grid
    data_index = np.random.randint(0, len(group['data']))
    data = group['data'][data_index]
    data_np = np.array(data, dtype=int)

    # randomize
    data_randomized = randomize_data(data_np)
    return data_randomized, label
