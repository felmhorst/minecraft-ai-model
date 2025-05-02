
## Data Cleaning
NBT data can be converted to JSON for inspection: `nbt_to_json`

All training data should be in the format:

```json
{
  "Schematic": {
    "Width": 16,
    "Height": 16,
    "Length": 16,
    "Offset": [0, 0, 0],
    "Blocks": {
      "Palette": {...},
      "Data": [...],
      "BlockEntities": []
    }
  }
}
```
Training:
- create global palette (for consistent block types)
- normalize palettes using global palette
- use `Schematic.Blocks.Data` as training data
Generation:
- generate `Schematic.Blocks.Data`
- build .schem file
- reduce palette IDs

Improvements:
- rotate structures for variation

## Palette Creation
- get minecraft data from `%appdata%\.minecraft\versions\<version>\<version>.jar`
- unzip
- open `./assets/minecraft/blockstates`

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
  - age= 0/1
  - leaves= large/small
  - waterlogged

Problems:
- wall_sign "facing" missing
- add waterlogged
- when.AND (e.g. chiseled_bookshelf)