from nbtlib import Compound, File, Int, List, Short, ByteArray, IntArray


def generate_sample_schematic():
    schematic = Compound({
        "Schematic": Compound({
            "Version": Int(3),
            "DataVersion": Int(4325),

            'Width': Short(1),
            'Height': Short(1),
            'Length': Short(1),
            'Offset': IntArray([0, 0, 0]),
            'Blocks': Compound({
                'Palette': Compound({
                    "minecraft:magenta_concrete": Int(0)
                }),
                "Data": ByteArray([0]),  # needs LongArray() if palette > 256 ?
                "BlockEntities": List[Compound]([])
            })
        })
    })
    return schematic
