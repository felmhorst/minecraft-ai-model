import json

import numpy as np
from nbtlib import Compound, File, Int, List, Short, ByteArray, IntArray

from config.colors import COLOR_WARNING, COLOR_DEFAULT
from config.paths import OUTPUT_PATH, BLOCK_MAPPING_ID_TO_TYPE, GLOBAL_BLOCK_PALETTE
from scripts.conversion.palette_conversion import to_local_palette
from scripts.conversion.array_conversion import convert_3d_data_to_1d
from scripts.ml.train_gan_embed_textures import sample_gan
from scripts.postprocessing.postprocess_schematic import postprocess_schematic


def data_3d_to_schematic(data_3d):
    shape = data_3d.shape
    data_flat = convert_3d_data_to_1d(data_3d)
    return to_schematic_file(data_flat, shape[2], shape[0], shape[1])


def to_schematic(data_flat, palette, w=16, h=16, l=16):
    """turns the data and palette into a schematic"""
    nbt_palette = Compound({key: Int(value) for key, value in palette.items()})
    print(f"{COLOR_WARNING}Warning: Clipping block palette!{COLOR_DEFAULT}")
    data_flat = np.clip(data_flat, 0, 127)  # todo: handle bigger block palettes
    nbt_data = ByteArray(data_flat)

    schematic = Compound({
        "Schematic": Compound({
            "Version": Int(3),
            "DataVersion": Int(4325),

            'Width': Short(w),
            'Height': Short(h),
            'Length': Short(l),
            'Offset': IntArray([0, 0, 0]),
            'Blocks': Compound({
                'Palette': nbt_palette,
                "Data": nbt_data,
                "BlockEntities": List[Compound]([])
            })
        })
    })
    return schematic


def get_sample_schematic_file():
    """creates a 1x1x1 sample schematic with a magenta_concrete block"""
    data = [0]
    palette = {"minecraft:magenta_concrete": 0}
    schem = to_schematic(data, palette, 1, 1, 1)
    schem_file = File(schem, root_name='Schematic')
    return schem_file


def to_schematic_file(global_data, w=16, h=16, l=16):
    """turns the data into a schematic file"""
    local_data, local_palette = to_local_palette(global_data)
    schem = to_schematic(local_data, local_palette, w, h, l)
    nbt_file = File(schem, root_name='Schematic')
    return nbt_file


def is_valid_data(data, palette):
    """returns true, if the data is valid for the given palette"""
    unique_ids = list(set(data))
    for id in unique_ids:
        is_valid = False
        if id < 0:
            print(f'Invalid id {id}')
            return False
        for block, block_id in palette.items():
            if id == block_id:
                is_valid = True
                break
        if not is_valid:
            print(f'invalid id {id}')
            return False
    return True


MAX_ID = 1105  # todo: calculate based on global palette


def generate_schematic(input_label: str):
    """generates a schematic and saves it to data/output"""
    data_norm = sample_gan(input_label)
    save_as_schematic(data_norm, OUTPUT_PATH)
    # data_flat = convert_3d_data_to_1d(data_3d)
    # schematic = to_schematic_file(data_flat)
    # schematic.save(output_path, gzipped=True)
    print(f'Generated schematic to {OUTPUT_PATH}')


def save_as_schematic(data_3d, output_path):
    shape = data_3d.shape

    with GLOBAL_BLOCK_PALETTE.open("r") as file:
        palette = json.load(file)

    data_3d, palette = postprocess_schematic(data_3d, palette)

    data_flat = convert_3d_data_to_1d(data_3d)
    schematic = to_schematic(data_flat, palette, shape[2], shape[0], shape[1])
    schematic_file = File(schematic, root_name='Schematic')
    schematic_file.save(output_path, gzipped=True)
