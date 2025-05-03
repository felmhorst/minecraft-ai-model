import json
from nbtlib import load
from scripts.conversion.nbt_conversion import nbt_to_dict
from pathlib import Path

base_path = Path(__file__).parent
global_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'palette.json'
global_reverse_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'palette-reverse.json'
WARNING = '\033[93m'
DEFAULT = '\033[0m'


def generate_palette_mapping(local_palette):
    palette_map = {}
    with open(global_palette_path, 'r') as file:
        global_palette = json.load(file)
        for key, local_id in local_palette.items():
            if key not in global_palette:
                print(f"{WARNING}Warning: Block '{key}' not found{DEFAULT}")
                continue
            global_id = global_palette[key]
            palette_map[local_id] = global_id
    return palette_map


def to_global_palette(nbt_file):
    schematic = load(nbt_file)
    blocks = schematic['Schematic']['Blocks']
    local_data = nbt_to_dict(blocks['Data'])
    local_palette = nbt_to_dict(blocks['Palette'])

    palette_mapping = generate_palette_mapping(local_palette)
    fallback_id = 3606

    global_data = [palette_mapping.get(n, fallback_id) for n in local_data]
    return global_data


def to_local_palette(global_data):
    local_palette = {}
    palette_mapping = {}
    global_data_set = list(set(global_data))
    with open(global_reverse_palette_path, 'r') as file:
        global_palette_reverse = json.load(file)
        for local_id in range(len(global_data_set)):
            global_id = global_data_set[local_id]
            key = global_palette_reverse[str(global_id)]
            palette_mapping[global_id] = local_id
            local_palette[key] = local_id
        # fallback_id = len(global_data_set)
        # local_palette['minecraft:air'] = fallback_id  # fallback
    local_data = [palette_mapping.get(n, 0) for n in global_data]  # todo: handle unknown
    return local_data, local_palette