from scripts.postprocessing.adjust_block_states import adjust_block_states
import numpy as np


def postprocess_schematic(
        data: np.ndarray,
        palette: dict
) -> tuple[np.ndarray, dict]:
    """
    Post-processes the data and palette to adjust block states.
    :param data: The voxel data
    :param palette: The block palette
    :return: The modified data and palette
    """
    return adjust_block_states(data, palette)
