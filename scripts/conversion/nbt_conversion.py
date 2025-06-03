import json
from nbtlib import load, tag


def nbt_to_dict(nbt_data):
    """converts NBT data to a python dictionary."""
    if isinstance(nbt_data, tag.Compound):
        return {key: nbt_to_dict(value) for key, value in nbt_data.items()}
    elif isinstance(nbt_data, tag.List):
        return [nbt_to_dict(item) for item in nbt_data]
    elif isinstance(nbt_data, (tag.ByteArray, tag.IntArray, tag.LongArray)):
        return list(nbt_data)
    elif isinstance(nbt_data, (tag.Byte, tag.Short, tag.Int, tag.Long)):
        return int(nbt_data)
    elif isinstance(nbt_data, (tag.Float, tag.Double)):
        return float(nbt_data)
    elif isinstance(nbt_data, (int, float, str, list, dict)):
        return nbt_data
    return str(nbt_data)


def nbt_to_json(nbt_file, output_file):
    """loads a NBT file, converts it to JSON and saves it."""
    schematic = load(nbt_file)
    schematic_dict = nbt_to_dict(schematic)

    with open(output_file, 'w') as f:
        json.dump(schematic_dict, f, indent=4)
    print(f"Conversion complete. Output saved to {output_file}.")
