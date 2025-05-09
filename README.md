# Minecraft Schematic Generator

## Project Structure
```
├── minecraft-ai/
│   ├── data/
│   │   ├── blockstates/  # default minecraft blockstates
│   │   ├── model/        # trained ML model & vocab
│   │   ├── output/       # generated files
│   │   ├── schematics/   # raw schematics (.nbt, .schem, .schematic)
│   │   └── training/     # training data
│   ├── scripts/
│   │   ├── conversion/   # data type & palette converisons
│   │   ├── ml/           # model training & prediction
│   │   └── palette/      # block palette
│   └── main.py
```

**Notes:**
- blockstates are taken from `%appdata%\.minecraft\versions\<version>\<version>.jar\assets\minecraft\blockstates` (jar must be unzipped)

## Getting Data
- WorldEdit:
  - mark a section - `//schem copy` - `//schem save <name>`
  - find the .schem in: `<paper-server-path>\plugins\WorldEdit\Schematics`
- Minecraft Structure Block:
  - save a structure with the Structure Block
  - find the .nbt in:
    - Singleplayer: `%appdata%\.minecraft\saves\<world-name>\structures`
    - Server: `<paper-server-path>\world\generated\minecraft\structures`

## Data Cleaning
NBT data (.nbt, .schem, .schematic) can be converted to JSON for inspection: `nbt_to_json`

Schem:
```json
{
  "Schematic": {
    "Version": 3,
    "DataVersion": 4325,
    "Metadata": {...}
    "Width": 16,
    "Height": 16,
    "Length": 16,
    "Offset": [0, 0, 0],
    "Blocks": {
      "Palette": {
        "minecraft:stripped_spruce_log[axis=y]": 0
      },
      "Data": [
        0
      ],
      "BlockEntities": []
    }
  }
}
```

NBT:
```json
{
  "size": [16, 16, 16],
  "DataVersion": 4325,
  "blocks": [
    {
      "state": 0,
      "pos": [4, 0, 5]
    }
  ],
  "palette": [
    {
      "Name": "minecraft:stripped_spruce_log",
      "Properties": {
        "axis": "y"
      }
    }

    }  ],
  "entities": []
}
```

Training:
- create global palette (for consistent block types)
- normalize palettes using global palette
- use `Schematic.Blocks.Data` as training data
- make all leaves persistent

Generation:
- generate `Schematic.Blocks.Data`
- build .schem file
- reduce palette IDs

Improvements:
- rotate structures for variation

## Palette Creation
- get minecraft data from `%appdata%\.minecraft\versions\<version>\<version>.jar\assets\minecraft\blockstates` (jar must be unzipped)

States:
- Minecraft Defaults:
  - face = ceiling/floor/wall
  - facing = north/east/south/west
  - powered = false/true
  - axis = x/y/z
  - shape = straight/inner_left/inner_right/outer_left/outer_right
  - half = bottom/top
- Schematic:
  - east=true,north=false,south=false,west=true (for multipart)
  - age= 0/1/2/3/4/5
  - leaves= large/small
  - waterlogged= true/false

Problems:
- wall_sign "facing" missing
- add waterlogged
- add powered
- when.AND (e.g. chiseled_bookshelf)
- leaves (persistent, distance)