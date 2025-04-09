import pickle
import os

def save_to_file(metadata, data):
    filename = metadata["filename"]
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{filename}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(metadata):
    filename = metadata["filename"]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{filename}.pkl"), "rb") as f:
        data = pickle.load(f)
    return data