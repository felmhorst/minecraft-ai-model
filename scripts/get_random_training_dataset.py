from scripts.get_random_training_data import get_random_training_data


def get_random_training_dataset(size):
    inputs = []
    outputs = []
    for i in range(size):
        '''shapes = {
            "cuboid": generate_random_cuboid,
            "pyramid": generate_random_pyramid,
            "sphere": generate_random_sphere,
        }
        shape = random.choice(["cuboid", "pyramid"])

        generate_shape = shapes[shape]
        hollow = random.choice([True, False])

        int_array = generate_shape(hollow=hollow)
        float_array = int_array.astype(np.float32)

        label = f"{'hollow' if hollow else 'solid'} {shape}"'''
        label, data = get_random_training_data()
        inputs.append(label)
        outputs.append(data)
    return inputs, outputs
