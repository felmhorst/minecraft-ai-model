from scripts.conversion.array_conversion import convert_1d_data_to_3d_array, convert_3d_data_to_1d
from scripts.conversion.palette_conversion import to_global_palette
from scripts.generate_schematic import generate_schematic, to_schematic_file, data_3d_to_schematic
from scripts.generate_training_data import generate_training_data
from scripts.ml.train import train_ml_model
from scripts.transformations.transform import flip_data_x, flip_data_z, flip_data_y, rotate_data

# convert nbt to json
# nbt_to_json("data/schematics/house_1.nbt", "data/house_1-nbt.json")

# generate global palette
# generate_block_palette('data/blockstates')

# converting a .schem to data (following the global palette)
# global_data = to_global_palette("data/schematics/house_1.schem")
# global_data_3d = convert_data_to_3d_array(global_data, 16, 16, 16)
# save_training_data(['house'], [global_data_3d])

# creating a .schem file from data
# nbt_file = convert_to_schematic_file(global_data)
# nbt_file.save('data/output/ai_house_25-05-02.schem', gzipped=True)

# generate_schem('house')
# nbt_to_json("data/output/ai_house_25-05-02.schem", "data/output/example.json")

# schematic = generate_sample_schematic_file()
# schematic.save('data/output/ai_house_25-05-02.schem', gzipped=True)

# convert_schematics_to_training_data('data/schematics')

# 1. generate training data
# generate_training_data('data/base-schematics')

# 2. train model
# train_ml_model()

# 3. generate schematic
# generate_schematic("house")


# with open('data/base-schematics/base_1_shape_6_roof_1.schem', 'r') as file:
from scripts.transformations.translate import translate
from scripts.transformations.trim import trim

"""global_data = to_global_palette('data/base-schematics/base_1_shape_6_roof_1.schem')
global_data_3d = convert_1d_data_to_3d_array(global_data, 16, 16, 16)
data_transformed = rotate_data(global_data_3d, 90)
data_flat = convert_3d_data_to_1d(data_transformed)
schematic = to_schematic_file(data_flat)
schematic.save('data/output/transform.schem', gzipped=True)
print(f'Generated transformed schematic.')
"""

global_data = to_global_palette('data/schematics/test.schem')
global_data_3d = convert_1d_data_to_3d_array(global_data, 16, 16, 16)
modified_data = translate(trim(global_data_3d), 1, 2, 3)
schematic = data_3d_to_schematic(modified_data)
schematic.save('data/output/transform.schem', gzipped=True)
