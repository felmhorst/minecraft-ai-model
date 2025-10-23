import json
from config.colors import COLOR_WARNING, COLOR_DEFAULT
from config.paths import BLOCK_MAPPING_TYPE_TO_ID, BLOCK_MAPPING_ID_TO_TYPE

FALLBACK_ID = 0


def get_palette_mapping(
        local_palette: dict
) -> dict:
    """returns a dict that maps each id in the local palette to an id in the global palette"""
    palette_mapping = {}
    with open(BLOCK_MAPPING_TYPE_TO_ID, 'r') as file:
        global_palette = json.load(file)
        for block_string, local_id in local_palette.items():
            if block_string not in global_palette:
                print(f"{COLOR_WARNING}Warning: Unknown block '{block_string}'. {COLOR_DEFAULT}")
                palette_mapping[local_id] = FALLBACK_ID
                continue
            palette_mapping[local_id] = global_palette[block_string]
    return palette_mapping


def to_global_palette(
        local_data: list,
        local_palette: dict
) -> list:
    """converts data + palette to normalized data"""
    palette_mapping = get_palette_mapping(local_palette)
    global_data = [palette_mapping.get(n, FALLBACK_ID) for n in local_data]
    return global_data


def to_local_palette(global_data):
    local_palette = {}
    palette_mapping = {}
    global_data_set = list(set(global_data))
    with open(BLOCK_MAPPING_ID_TO_TYPE, 'r') as file:
        global_palette_reverse = json.load(file)
        for local_id in range(len(global_data_set)):
            global_id = global_data_set[local_id]
            block_string = global_palette_reverse[str(global_id)]["name"]
            palette_mapping[global_id] = local_id
            local_palette[block_string] = local_id
    local_data = [palette_mapping.get(n, FALLBACK_ID) for n in global_data]  # todo: always set 0 to air
    return local_data, local_palette
