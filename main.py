import torch

from scripts.conversion.nbt_conversion import nbt_to_json
from scripts.generate_schematic import generate_schematic
# from scripts.ml.train_gan import train_gan, continue_training_gan
from scripts.ml.train_gan_embed_textures import train_gan, continue_training_gan

# check CUDA
print("torch version:", torch.__version__)
print("expected CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("using cuda", torch.cuda.get_device_name(0))

"""label, data3d = get_random_training_data()
print(label)

data_flat = convert_3d_data_to_1d(data3d)
local_data, local_palette = to_local_palette(data_flat)
populated_data, populated_palette = populate_block_properties(local_data, local_palette)

schematic = to_schematic(populated_data, populated_palette)
schematic_file = File(schematic, root_name='Schematic')
schematic_file.save("data/output/test.schem", gzipped=True)
print('saved schematic.')"""

# train_gan()
# continue_training_gan()
generate_schematic('gable house')
nbt_to_json('data/output/generated.schem', 'data/output/generated.json')
