import json

import numpy as np

from config.paths import TRAINING_SCHEMATICS_LIST, TRAINING_DATA_LIST, SCHEMATICS_DIR
from scripts.conversion.array_conversion import convert_1d_data_to_3d
from scripts.conversion.palette_conversion import to_global_palette
from scripts.palette.strip_block_properties import strip_block_properties
from scripts.schematic.load_schematic import load_schematic


def prepare_training_data():
    """converts the training data from schematics into normalized numeric data"""
    data_groups = []
    with open(TRAINING_SCHEMATICS_LIST, 'r') as file:
        schematic_list = json.load(file)
        for group in schematic_list['groups']:
            data_arr = []
            for file_name in group['files']:
                data_arr.append(prepare_training_data_array(file_name))
            data_groups.append({
                "labels": group['labels'],
                "data": data_arr
            })

    # save training data
    with open(TRAINING_DATA_LIST, 'w') as file:
        training_data = {"groups": data_groups}
        json.dump(training_data, file)


def prepare_training_data_array(
        file_name: str
) -> list:
    local_data, local_palette = load_schematic(SCHEMATICS_DIR / file_name)
    stripped_data, stripped_palette = strip_block_properties(local_data, local_palette)
    global_data = to_global_palette(stripped_data, stripped_palette)
    global_data3d = convert_1d_data_to_3d(global_data).tolist()
    return global_data3d


def load_training_data():
    with open(TRAINING_DATA_LIST, 'r') as file:
        data_groups = json.load(file)["groups"]
        return data_groups
