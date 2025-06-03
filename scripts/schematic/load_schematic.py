from nbtlib import load
from scripts.conversion.nbt_conversion import nbt_to_dict


def load_schematic(file_path):
    schematic = load(file_path)
    width = schematic['Schematic']['Width']
    length = schematic['Schematic']['Length']
    height = schematic['Schematic']['Height']
    if width != 16 or length != 16 or height != 16:
        print(f'Warning: Size mismatch. ({width}, {height}, {length})')
    blocks = schematic['Schematic']['Blocks']
    local_data = nbt_to_dict(blocks['Data'])
    local_palette = nbt_to_dict(blocks['Palette'])
    return local_data, local_palette
