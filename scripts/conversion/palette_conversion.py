import json
from nbtlib import load

from scripts.conversion.block_representation_conversion import block_string_to_dict
from scripts.conversion.nbt_conversion import nbt_to_dict
from pathlib import Path

base_path = Path(__file__).parent
global_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'block_types.json'
global_reverse_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'block_types_reverse.json'
WARNING = '\033[93m'
DEFAULT = '\033[0m'

FALLBACK_ID = 0


def generate_palette_mapping(local_palette):
    """returns a dict that maps each id in the local palette to an id in the global palette"""
    palette_map = {}
    with open(global_palette_path, 'r') as file:
        global_palette = json.load(file)
        for block_string, local_id in local_palette.items():
            block_dict = block_string_to_dict(block_string)
            if block_dict["type"] not in global_palette:
                print(f"{WARNING}Warning: Block '{block_dict['type']}' not found{DEFAULT}")
                continue
            global_id = global_palette[block_dict["type"]]
            palette_map[local_id] = global_id
    return palette_map


def to_global_palette(nbt_file_path):
    schematic = load(nbt_file_path)
    blocks = schematic['Schematic']['Blocks']
    local_data = nbt_to_dict(blocks['Data'])
    local_palette = nbt_to_dict(blocks['Palette'])

    palette_mapping = generate_palette_mapping(local_palette)

    global_data = [palette_mapping.get(n, FALLBACK_ID) for n in local_data]
    return global_data


def to_local_palette(global_data):
    local_palette = {}
    palette_mapping = {}
    global_data_set = list(set(global_data))
    with open(global_reverse_palette_path, 'r') as file:
        global_palette_reverse = json.load(file)
        for local_id in range(len(global_data_set)):
            global_id = global_data_set[local_id]
            block_string = global_palette_reverse[str(global_id)]
            palette_mapping[global_id] = local_id
            local_palette[block_string] = local_id
    local_data = [palette_mapping.get(n, FALLBACK_ID) for n in global_data]  # todo: always set 0 to air
    return local_data, local_palette
