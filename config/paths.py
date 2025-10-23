from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = DATA_DIR / "output"

# block palette
PALETTE_DIR = DATA_DIR / "palette"
BLOCK_WHITELIST = PALETTE_DIR / "block_whitelist.json"
GLOBAL_BLOCK_PALETTE = PALETTE_DIR / "block_palette.json"
BLOCK_MAPPING_TYPE_TO_ID = PALETTE_DIR / "block_type_to_id.json"
BLOCK_MAPPING_ID_TO_TYPE = PALETTE_DIR / "block_id_to_type.json"

# training data
TRAINING_DIR = DATA_DIR / "training"
SCHEMATICS_DIR = TRAINING_DIR / "schematics"
TRAINING_SCHEMATICS_LIST = TRAINING_DIR / "schematics.json"
TRAINING_DATA_LIST = TRAINING_DIR / "training_data.json"

# output
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "generated.schem"

# checkpoints
CHECKPOINT_DIR = DATA_DIR / "model"
