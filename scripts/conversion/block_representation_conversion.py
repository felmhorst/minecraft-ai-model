
def block_string_to_dict(block_string):
    """converts a block string in the format "minecraft:<block>[<properties>] into a dictionary."""
    block_dict = {}
    parts = block_string.split("[")
    block_dict["type"] = parts[0].split(":")[1]
    if len(parts) == 1:
        return block_dict
    properties = parts[1][:-1].split(",")
    for prop in properties:
        parts = prop.split("=")
        block_dict[parts[0]] = parts[1]
    return block_dict


def block_dict_to_string(block_dict):
    """converts a dictionary that describes a block into a string."""
    properties = []
    for key, value in block_dict:
        if key == "type":
            continue
        properties.append(f"{key}={value}")
    properties_string = ",".join(properties)
    block_string = f"minecraft:{block_dict['type']}[{properties_string}]"
    return block_string
