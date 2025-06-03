
def validate_data_matches_palette(data, palette):
    """returns True, if the data matches the palette."""
    valid_ids = palette.values()
    for id in set(data):
        if id not in valid_ids:
            print(f"Invalid id '{id}'")
            return False
    return True
