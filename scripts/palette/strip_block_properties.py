

def map_data(
        data: list,
        mapping: dict
) -> list:
    """maps the data using the mapping provided"""
    new_data = []
    for id in data:
        if id in mapping.keys():
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

    new_data = map_data(data, palette_mapping)
    return new_data, new_palette
