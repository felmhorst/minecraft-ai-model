from pathlib import Path
from scripts.conversion.palette_conversion import to_global_palette_from_file
from scripts.conversion.array_conversion import convert_1d_data_to_3d
from scripts.training_data import save_training_data


def generate_training_data(folder_path):
    """generates a training data set"""
    input_text = []
    output = []
    folder = Path(folder_path)
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        global_data = to_global_palette_from_file(file_path)
        global_data_3d = convert_1d_data_to_3d(global_data, 16, 16, 16)
        output.append(global_data_3d)
        input_text.append("house")
    save_training_data(input_text, output)
