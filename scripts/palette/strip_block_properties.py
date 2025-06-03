from pathlib import Path
import json
from scripts.palette.block_representation_conversion import block_dict_to_string

base_path = Path(__file__).parent
global_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'block_type_to_dict.json'


def apply_palette_mapping(data, mapping):
    new_data = []
    for id in data:
        if id in mapping:
            new_data.append(mapping[id])
        else:
            new_data.append(id)
    return new_data


def strip_block_properties(data, palette):
    """removes all block properties in a palette"""
    new_palette = {}
    palette_mapping = {}
    i = 0
    for key, value in palette.items():
        stripped_key = key.split("[")[0].split(":")[1]
        if stripped_key not in new_palette.keys():
            new_palette[stripped_key] = i
            palette_mapping[value] = i
            i += 1
        else:
            palette_mapping[value] = new_palette[stripped_key]

    new_data = apply_palette_mapping(data, palette_mapping)
    return new_data, new_palette


def populate_block_properties(data, palette):
    with open(global_palette_path, 'r') as file:
        block_type_to_dict = json.load(file)
        new_palette = {'minecraft:air': 0}
        palette_mapping = {}
        i = 0
        for key, value in palette.items():
            if key in new_palette.keys():
                continue
            if key not in block_type_to_dict:
                palette_mapping[value] = 0
                continue
            block_dict = block_type_to_dict[key]
            block_str = block_dict_to_string(block_dict)
            new_palette[block_str] = i
            palette_mapping[value] = i
            i += 1
    new_data = apply_palette_mapping(data, palette_mapping)

    return new_data, new_palette
