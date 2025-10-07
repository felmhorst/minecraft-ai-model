# Training Data
This file documents types and variations of training data, how they are obtained and used.

## File Types
The following file types are associated with minecraft structures:
- `.nbt` (Named Binary Tag) - used by Minecraft Java Edition, see [NBT format on Minecraft Wiki](https://minecraft.fandom.com/wiki/NBT_format)
- `.snbt` - stringified NBT for readability
- `.mcstructure` - used by Minecraft Bedrock Edition
- `.schematic` - used by WorldEdit < 1.13, formatted in NBT
- `.schem` - used by WorldEdit >= 1.13, formatted in NBT

## Voxel Grid
Independent of the file type, the data is typically structured as a flat array, which represents a 3d array of the
structure `[y, z, x]` that begins in the bottom north-west corner. This results in the following rules:
- `y`: top > bottom
- `z`: south > north
- `x`: east > west

## Block Representations
Because the majority of blocks in a minecraft structure are air, training a model on pure texture ids is likely to
result in a mode collapse. To prevent this, the data is split into occupancy (0: air, 1: solid block) and texture
(0-n: solid block id).
**Occupancy**:
- **Probabilities** (0-1): representing the occupancy in the training data as 0 or 1 and generating probabilities causes
  issues, because this is not a linear correlation and is thus difficult to learn for the model
- **Distance Field**: encoding distance to the closest solid block makes the data distribution linear.
  - Signed Distance Field (SDF): not useful for Minecraft structures, since there is no clear inside/outside
  - Unsigned Distance Field (UDF): useful for Minecraft structures
  - Truncated Distance Field (TDF): can either be signed or unsigned. Typically used for sparse voxel grids, as is the
    case here.

## Generate Training Data
Training data could either be generated automatically (for simple structures), built manually, or scraped.
Basic shapes can be used in early stages to help understand basic concepts like shapes, walls, or hollowness.
Basic structures are great to introduce more complex concepts that are easy to augment.
- **Basic Shapes** (see `/scripts/generation`)
  - cuboid
  - pyramid
  - sphere
- **Basic Structures**
  - buildings (house, tower, pyramid, temple)
  - decoration (bench, tent, well, shed, vehicle, lantern, statue, chimney, shelf, bridge, mine, planter, market stand,
    street, wagon, barrel, crane, pipeline, fountain, logs, portal, canon, air balloon, smeltery, enchantment table,
    boat)
  - nature (tree, mushroom, rock, pond)

## Data Augmentation
Data augmentation is used to increase the variety of training data and help with generalization.
- add small noise
- transformations (see `/scripts/transformations`)
  - flip
  - rotate
  - translate
- replace/jitter material

## Labels
These are guidelines for writing labels for the training data to optimise generalization.
- write 5-15 word phrases (e.g. small medieval stone house with a steep wooden roof and a chimney)
- write 3-5 variations for language augmentation
- describe the following aspects:
  - structure & parts (e.g. house, roof)
  - material (e.g. wood, stone, oak)
  - shape (e.g. steep, gentle, round)
  - styles (e.g. modern, medieval, fantasy)
  - color (e.g. yellow, dark)
  - size (e.g. small, large, long, tall)
  - count

---

# Postprocessing
- replace floating blocks with air (no direct neighbours)
- adjust multi-blocks by settings adjacent blocks (e.g. door)
- adjust block states
  - half: prefer side with a solid block (e.g. slab, stairs)
  - axis: prefer axis with the same blocks (e.g. log)
  - facing: prefer side with a solid block (e.g. stairs)
  - shape: adjust based on surrounding blocks
- jittering? (e.g. plants, moss/cracked block variants)