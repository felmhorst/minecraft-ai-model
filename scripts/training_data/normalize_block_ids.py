import json
from pathlib import Path

base_path = Path(__file__).parent
global_palette_path = base_path / '..' / '..' / 'data' / 'palette' / 'block_type_to_id.json'


def get_max_block_id():
    with open(global_palette_path, 'r') as file:
        block_ids = json.load(file)
        return len(block_ids) - 1


MAX_BLOCK_ID = get_max_block_id()


def normalize_block_ids(data):
    return data / MAX_BLOCK_ID


def denormalize_block_ids(data):
    denormalized_data = (data * MAX_BLOCK_ID).round().astype(int)
    return denormalized_data
