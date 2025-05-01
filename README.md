
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