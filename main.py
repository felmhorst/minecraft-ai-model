from scripts.conversion.array_conversion import convert_1d_data_to_3d_array
from scripts.conversion.palette_conversion import to_global_palette
from scripts.generate_schematic import generate_schematic, to_schematic_file, data_3d_to_schematic, save_as_schematic
from scripts.generate_training_data import generate_training_data
from scripts.generation.generate_cuboid import generate_random_cuboid
from scripts.generation.generate_pyramid import generate_random_pyramid
from scripts.generation.generate_sphere import generate_random_sphere
from scripts.ml.train_diffusion_model import train_latent_diffusion_model
from scripts.ml.train_gan import train_gan, continue_training_gan
from scripts.ml.train_vae import train
from scripts.palette.generate_global_palette import save_block_types
from scripts.transformations.randomize_data import randomize_data, get_random_data, get_random_dataset

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

# train_diffusion_model()
# continue_training_diffusion_model()
# train_latent_diffusion_model()
# generate_schematic()

# train_gan()
generate_schematic("solid pyramid")
continue_training_gan()
# generate_schematic("solid cuboid")

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

"""
global_data = to_global_palette('data/base-schematics/base_1_shape_1_roof_1.schem')
global_data_3d = convert_1d_data_to_3d_array(global_data, 16, 16, 16)
# modified_data = randomize_data(global_data_3d)
schematic = data_3d_to_schematic(global_data_3d)
schematic.save('data/output/transform.schem', gzipped=True)"""

# data_3d = get_random_data()
# schematic = data_3d_to_schematic(data_3d)
# schematic.save('data/output/transform.schem', gzipped=True)

# save_block_types('data/blockstates')

# data_3d = get_random_dataset(1)[0]
# save_as_schematic(data_3d, "data/output/sphere.schem")

# save_as_schematic(get_random_dataset(1)[0], "data/output/dirt.schem")