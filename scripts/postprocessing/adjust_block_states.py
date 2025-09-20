import numpy as np


def adjust_block_states(
        data: np.ndarray,
        palette: dict
) -> tuple[np.ndarray, dict]:
    """
    Automatically adjusts the block states, in particular the following ones:
        - half (top/bottom): prefers the side with a neighbouring solid block.
        - type (top/bottom): prefers the side with a neighbouring solid block.
        - axis (x/y/z): prefers the axis along which the most direct neighbours of the same block type are found
        - facing (north/east/south/west): prefers the side with a neighbouring solid block.
        - shape (straight/inner_left/inner_right/outer_left/outer_right): is adjusted based on surrounding blocks
    :param data: The voxel data
    :param palette: The block palette
    :return: The modified data and palette
    """
    data, palette = adjust_block_state_half(data, palette)
    data, palette = adjust_block_state_type(data, palette)
    data, palette = adjust_block_state_facing(data, palette)
    return data, palette


def adjust_block_state_half(
    data_3d: np.ndarray,
    palette: dict
) -> tuple[np.ndarray, dict]:
    """
    Automatically adjusts the block state "half".
    :param data_3d: The voxel data
    :param palette: The block palette
    :return: The modified data and palette
    """

    new_palette = palette.copy()
    new_data = data_3d.copy()
    max_id = len(palette) - 1

    for block_name, block_id in palette.items():
        if "half=bottom" in block_name:
            # add variant with half=top to palette
            block_top_variant = block_name.replace("half=bottom", "half=top")
            max_id += 1
            new_palette[block_top_variant] = max_id

            # mask blocks to replace
            mask = np.zeros_like(new_data, dtype=bool)
            mask[1:-1] = new_data[1:-1] == block_id
            condition = (new_data[:-2] == 0) & (new_data[2:] > 0)
            mask[1:-1] &= condition

            # set block ids for half=top
            new_data[mask] = max_id

    return new_data, new_palette


def adjust_block_state_type(
    data_3d: np.ndarray,
    palette: dict
) -> tuple[np.ndarray, dict]:
    """
    Automatically adjusts the block state "type".
    :param data_3d: The voxel data
    :param palette: The block palette
    :return: The modified data and palette
    """

    new_palette = palette.copy()
    new_data = data_3d.copy()
    max_id = len(palette) - 1

    for block_name, block_id in palette.items():
        if "type=bottom" in block_name:
            # add variant with half=top to palette
            block_top_variant = block_name.replace("type=bottom", "type=top")
            max_id += 1
            new_palette[block_top_variant] = max_id

            # mask blocks to replace
            mask = np.zeros_like(new_data, dtype=bool)
            mask[1:-1] = new_data[1:-1] == block_id
            condition = (new_data[:-2] == 0) & (new_data[2:] > 0)
            mask[1:-1] &= condition

            # set block ids for half=top
            new_data[mask] = max_id

    return new_data, new_palette


def adjust_block_state_facing(
    data_3d: np.ndarray,
    palette: dict
) -> tuple[np.ndarray, dict]:
    """
    Automatically adjusts the block state "facing".
    :param data_3d: The voxel data [y, z, x]
    :param palette: The block palette (block_name -> block_id)
    :return: The modified data and palette
    """

    new_palette = palette.copy()
    new_data = data_3d.copy()
    max_id = len(palette) - 1

    for block_name, block_id in palette.items():
        if "facing=east" in block_name:
            # add variants for other facings
            facing_map = {}
            for value in ("north", "south", "west"):
                block_variant = block_name.replace("facing=east", f"facing={value}")
                max_id += 1
                new_palette[block_variant] = max_id
                facing_map[value] = max_id
            # keep original east-facing id
            facing_map["east"] = block_id

            # mask with this block id
            block_mask = new_data == block_id

            # neighbor occupancy masks
            east_mask = np.zeros_like(new_data, dtype=bool)
            west_mask = np.zeros_like(new_data, dtype=bool)
            south_mask = np.zeros_like(new_data, dtype=bool)
            north_mask = np.zeros_like(new_data, dtype=bool)
            east_mask[:, :, :-1] = new_data[:, :, 1:] > 0
            west_mask[:, :, 1:] = new_data[:, :, :-1] > 0
            south_mask[:, :-1, :] = new_data[:, 1:, :] > 0
            north_mask[:, 1:, :] = new_data[:, :-1, :] > 0

            # neighbor counts
            neighbor_count = east_mask + west_mask + south_mask + north_mask

            # if there is a neighbor on one side, face that neighbor
            for direction, arr in zip(
                ("east", "west", "south", "north"),
                (east_mask, west_mask, south_mask, north_mask)
            ):
                cond = block_mask & (neighbor_count == 1) & arr
                new_data[cond] = facing_map[direction]

            # if there are neighbors one three sides, face opposite the empty side
            opposite = {"east": "west", "west": "east", "north": "south", "south": "north"}
            for direction, arr in zip(
                    ("east", "west", "south", "north"),
                    (east_mask, west_mask, south_mask, north_mask),
            ):
                cond = block_mask & (neighbor_count == 3) & (~arr)
                new_data[cond] = facing_map[opposite[direction]]

            # for direction, arr in zip(
            #     ("east", "west", "south", "north"),
            #     (east_mask, west_mask, south_mask, north_mask)
            # ):
            #     cond = block_mask & (neighbor_count == 3) & ~arr
            #     new_data[cond] = facing_map[direction]

            # if there are neighbors on two opposite sides, face either empty side
            cond = block_mask & (neighbor_count == 2) & east_mask & west_mask
            new_data[cond] = facing_map["south"]
            cond = block_mask & (neighbor_count == 2) & north_mask & south_mask
            new_data[cond] = facing_map["east"]

            # if there are neighbors on two connected sides, face either neighbor
            cond = block_mask & (neighbor_count == 2) & east_mask & north_mask
            new_data[cond] = facing_map["east"]
            cond = block_mask & (neighbor_count == 2) & east_mask & south_mask
            new_data[cond] = facing_map["east"]
            cond = block_mask & (neighbor_count == 2) & west_mask & north_mask
            new_data[cond] = facing_map["south"]  # arbitrary choice
            cond = block_mask & (neighbor_count == 2) & west_mask & south_mask
            new_data[cond] = facing_map["south"]

            # Rule 5: 0 or 4 sides -> keep default east
            cond = block_mask & ((neighbor_count == 0) | (neighbor_count == 4))
            new_data[cond] = facing_map["east"]

    return new_data, new_palette

