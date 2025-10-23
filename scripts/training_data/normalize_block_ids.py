import json
from config.paths import BLOCK_WHITELIST


def get_max_block_id():
    with open(BLOCK_WHITELIST, 'r') as file:
        blocks = json.load(file)
        return len(blocks) - 1


MAX_BLOCK_ID = get_max_block_id()


def normalize_block_ids(data):
    return data / MAX_BLOCK_ID


def denormalize_block_ids(data):
    denormalized_data = (data * MAX_BLOCK_ID).round().astype(int)
    return denormalized_data
