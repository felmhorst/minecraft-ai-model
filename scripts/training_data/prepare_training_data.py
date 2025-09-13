import json
from pathlib import Path

from scripts.conversion.array_conversion import convert_1d_data_to_3d
from scripts.conversion.palette_conversion import to_global_palette
from scripts.palette.strip_block_properties import strip_block_properties
from scripts.schematic.load_schematic import load_schematic

base_path = Path(__file__).parent
schematic_list_path = base_path / '..' / '..' / 'data' / 'training' / 'schematics.json'
training_data_path = base_path / '..' / '..' / 'data' / 'training' / 'training_data.json'
schematic_folder_path = base_path / '..' / '..' / 'data' / 'base-schematics'


def prepare_training_data():
    """converts the schematic files included in schematics.json into training data"""
    data_groups = []
    with open(schematic_list_path, 'r') as file:
        schematic_list = json.load(file)
        for group in schematic_list['groups']:
            labels = group['labels']
            files = group['files']
            data_arr = []
            for file_name in files:
                data_arr.append(prepare_training_data_array(file_name))
            data_groups.append({
                "labels": labels,
                "data": data_arr
            })

    # save training data
    with open(training_data_path, 'w') as file:
        training_data = {"groups": data_groups}
        json.dump(training_data, file)


def prepare_training_data_array(file_name):
    local_data, local_palette = load_schematic(schematic_folder_path / file_name)
    stripped_data, stripped_palette = strip_block_properties(local_data, local_palette)
    global_data = to_global_palette(stripped_data, stripped_palette)
    global_data3d = convert_1d_data_to_3d(global_data)
    global_data3d_list = global_data3d.tolist()
    return global_data3d_list


def load_training_data():
    with open(training_data_path, 'r') as file:
        data_groups = json.load(file)["groups"]
        return data_groups
