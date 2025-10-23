from pathlib import Path
from nbtlib import load

from config.colors import COLOR_WARNING, COLOR_DEFAULT
from scripts.conversion.nbt_conversion import nbt_to_dict


def load_schematic(
        file_path: str | Path
) -> tuple[list, dict]:
    """returns the data and palette of the schematic under the specified path"""
    schematic = load(file_path)
    width = schematic['Schematic']['Width']
    length = schematic['Schematic']['Length']
    height = schematic['Schematic']['Height']
    if width != 16 or length != 16 or height != 16:
        print(f'{COLOR_WARNING}Warning: Size mismatch. Expected: (16, 16, 16). Found: ({width}, {height}, {length}).'
              f'File: {file_path}{COLOR_DEFAULT}')
    blocks = schematic['Schematic']['Blocks']
    local_data = nbt_to_dict(blocks['Data'])
    local_palette = nbt_to_dict(blocks['Palette'])
    return local_data, local_palette
