import json
from pathlib import Path

base_path = Path(__file__).parent
training_file = base_path / '..' / 'data' / 'training' / 'training.json'


def save_training_data(text_input, output):
    """saves training data to data/training/training.json"""
    output_list = [data.tolist() for data in output]
    data = {
        "input": text_input,
        "output": output_list
    }
    example_count = len(output_list)
    with open(training_file, 'w') as file:
        json.dump(data, file)
    print(f'Saved dataset with {example_count} examples to {training_file}')


def load_training_data():
    """loads training data from data/training/training.json and returns it"""
    with open(training_file, 'r') as file:
        data = json.load(file)
        return data['input'], data['output']
