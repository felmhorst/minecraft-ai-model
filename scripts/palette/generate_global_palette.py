from pathlib import Path
import json


base_path = Path(__file__).parent
global_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'palette.json'
global_reverse_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'palette-reverse.json'


block_type_palette = base_path / '..' / '..' / 'data' / 'palette' / 'block_types.json'
block_type_palette_reverse = base_path / '..' / '..' / 'data' / 'palette' / 'block_types_reverse.json'


def get_property_combinations(d):
    def get_dict_without_key(d, key):
        d_copy = dict(d)
        d_copy.pop(key, None)
        return d_copy

    if not d:
        return ['']
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


def can_be_powered(block_type):
    return block_type.endswith(('_door', '_trapdoor'))


def can_be_waterlogged(block_type):
    return block_type.endswith(('_bars', '_fence', '_ladder', '_pane', '_slab', '_stairs', '_trapdoor', '_wall', '_wall_sign'))


def get_runtime_variants(block_type, block):
    runtime_variants = [block]

    if can_be_powered(block_type):
        variants_to_process = runtime_variants.copy()
        runtime_variants = []
        for variant in variants_to_process:
            base = variant[:-1] if variant.endswith(']') else variant
            runtime_variants.append(f'{base},powered=true]')
            runtime_variants.append(f'{base},powered=false]')

    if can_be_waterlogged(block_type):
        variants_to_process = runtime_variants.copy()
        runtime_variants = []
        for variant in variants_to_process:
            base = variant[:-1] if variant.endswith(']') else variant
            runtime_variants.append(f'{base},waterlogged=true]')
            runtime_variants.append(f'{base},waterlogged=false]')

    return runtime_variants


def get_block_configurations_from_variants(block_type, variants):
    configurations = []
    for config in variants.keys():
        if config != "":
            config = f'[{config}]'
        block = f'minecraft:{block_type}{config}'
        configurations += get_runtime_variants(block_type, block)
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


def generate_reverse_block_palette(palette):
    reverse_palette = {}
    for key, value in palette.items():
        reverse_palette[value] = key
    with open(global_reverse_palette_path, 'w') as file:
        json.dump(reverse_palette, file)


def generate_block_palette(blockstate_folder_path):
    """saves a flat block palette with unique ids to data/palette/palette.json"""
    palette = convert_blockstates_to_palette(blockstate_folder_path)
    with open(global_palette_path, 'w') as file:
        json.dump(palette, file)
    generate_reverse_block_palette(palette)
    print('Block palette saved to data/palette.json')


def get_block_types_from_folder(folder_path):
    """returns a list of block configs from a folder_path (/assets/minecraft/blockstates)"""
    folder = Path(folder_path)
    block_types = []
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        block_types.append(file_path.stem)
    return block_types


def save_block_types(blockstate_folder_path):
    folder = Path(blockstate_folder_path)

    # block_types = get_block_types_from_folder(blockstate_folder_path)
    block_type_dict = {"air": "0"}
    reverse_block_type_dict = {"0": "minecraft:air"}
    i = 1

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        block_type = file_path.stem
        if block_type == 'air':
            continue
        block_type_dict[block_type] = str(i)
        reverse_block_type_dict[str(i)] = get_block_string(file_path)
        i += 1
    with open(block_type_palette, 'w') as file:
        json.dump(block_type_dict, file)
    with open(block_type_palette_reverse, 'w') as file:
        json.dump(reverse_block_type_dict, file)


def get_block_string(file_path):
    with file_path.open('r') as file:
        data = json.load(file)
        if "variants" not in data:
            return f'minecraft:{file_path.stem}'
        variant = next(iter(data["variants"]))
        if variant == "":
            return f'minecraft:{file_path.stem}'
        if can_be_waterlogged(file_path.stem):
            return f'minecraft:{file_path.stem}[{variant},waterlogged=false]'
        return f'minecraft:{file_path.stem}[{variant}]'
