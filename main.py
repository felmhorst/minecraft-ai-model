from scripts.generate_schematic import generate_schematic
from scripts.generate_training_data import generate_training_data
from scripts.ml.train import train_ml_model

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
# generate_training_data('data/schematics')

# 2. train model
# train_ml_model()

# 3. generate schematic
generate_schematic("house")
