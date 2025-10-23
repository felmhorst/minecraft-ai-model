import json

from config.colors import COLOR_WARNING, COLOR_DEFAULT
from config.paths import BLOCK_WHITELIST, BLOCK_MAPPING_TYPE_TO_ID, BLOCK_MAPPING_ID_TO_TYPE, GLOBAL_BLOCK_PALETTE


def generate_block_mapping():
    """converts the block whitelist into mappings between type and id"""
    block_map_type_to_id = {}
    block_map_id_to_type = {}
    global_block_palette = {}
    with BLOCK_WHITELIST.open('r') as file:
        blocks = json.load(file)
        for i in range(len(blocks)):
            block = blocks[i]
            if block["name"] in block_map_type_to_id.keys():
                print(f"{COLOR_WARNING}Warning: Duplicate block '{block['name']}' in block_whitelist.json {COLOR_DEFAULT}")
                continue
            block_map_id_to_type[str(i)] = block
            block_map_type_to_id[block["name"]] = i
            block_name_with_properties = f"minecraft:{block['name']}{block['property_string']}"
            global_block_palette[block_name_with_properties] = i
    with BLOCK_MAPPING_TYPE_TO_ID.open('w') as file:
        json.dump(block_map_type_to_id, file)
    with BLOCK_MAPPING_ID_TO_TYPE.open('w') as file:
        json.dump(block_map_id_to_type, file)
    with GLOBAL_BLOCK_PALETTE.open('w') as file:
        json.dump(global_block_palette, file)
