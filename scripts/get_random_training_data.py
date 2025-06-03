import numpy as np
from scripts.normalize_block_ids import normalize_block_ids
from scripts.prepare_training_data import load_training_data
from scripts.transformations.randomize_data import randomize_data


def get_random_training_data():
    data_groups = load_training_data()

    # random group
    group_index = np.random.randint(0, len(data_groups))
    group = data_groups[group_index]

    # random label
    label_index = np.random.randint(0, len(group['labels']))
    label = group['labels'][label_index]

    # random (schematic) data
    data_index = np.random.randint(0, len(group['data']))
    data = group['data'][data_index]
    data_np = np.array(data, dtype=float)
    data_normalized = normalize_block_ids(data_np)

    # randomize
    data_randomized = randomize_data(data_normalized)
    return label, data_randomized
