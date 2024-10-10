import json
import pandas as pd

def load_data(filepath):
    """Load production data from a JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

