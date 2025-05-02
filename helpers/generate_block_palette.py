from pathlib import Path
import json


def get_property_combinations(d):

    def get_dict_without_key(d, key):
        d_copy = dict(d)
        d_copy.pop(key, None)
        return d_copy

    # base case
    if not d:
        return ['']

    # this iter
    key = next(iter(d))
    values = d[key]
    part_dict = get_dict_without_key(d, key)
    part_combinations = get_property_combinations(part_dict)
    combinations = []
    for value in values:
        for combination in part_combinations:
            if combination == '':
                combinations.append(f'{key}={value}')
            else:
                combinations.append(f'{key}={value},{combination}')
    return combinations


def get_block_configurations_from_variants(block_type, variants):
    configurations = []
    for config in variants.keys():
        if config != "":
            config = f'[{config}]'
        block = f'minecraft:{block_type}{config}'
        configurations.append(block)
    return configurations


def get_block_configurations_from_multipart(block_type, multipart):
    # collect properties & values
    properties = {}

    def add_properties(items):
        nonlocal properties
        for key, value in items:
            if key not in properties:
                properties[key] = []
            if value not in properties[key]:
                properties[key].append(value)

    for rule in multipart:
        if 'when' not in rule:
            continue
        elif 'AND' in rule['when']:
            for rule_part in rule['when']['AND']:
                add_properties(rule_part.items())
        elif 'OR' in rule['when']:
            for rule_part in rule['when']['OR']:
                add_properties(rule_part.items())
        else:
            add_properties(rule['when'].items())
    # create configurations
    combinations = get_property_combinations(properties)
    configurations = []
    for combination in combinations:
        block = f'minecraft:{block_type}[{combination}]'
        configurations.append(block)

    return configurations


def get_block_configurations_from_file(file_path):
    """returns a list of block configs from a file_path (/assets/minecraft/blockstates/<block>.json)"""
    file_name = file_path.stem
    with file_path.open('r') as file:
        data = json.load(file)  # variants or multipart
        if 'variants' in data:
            return get_block_configurations_from_variants(file_name, data['variants'])
        if 'multipart' in data:
            return get_block_configurations_from_multipart(file_name, data['multipart'])
    print(f'Unknown block configurations for block {file_name}')
    return []



def get_block_configurations_from_folder(folder_path):
    """returns a list of block configs from a folder_path (/assets/minecraft/blockstates)"""
    folder = Path(folder_path)
    configurations = []
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        configurations += get_block_configurations_from_file(file_path)
    return configurations


def convert_blockstates_to_palette(folder_path):
    configurations = get_block_configurations_from_folder(folder_path)
    palette = {}
    for i in range(0, len(configurations)):
        config = configurations[i]
        palette[config] = i
    return palette


def generate_block_palette(folder_path):
    palette = convert_blockstates_to_palette(folder_path)
    with open('data/palette.json', 'w') as file:
        json.dump(palette, file)
    print('Block palette saved to data/palette.json')
