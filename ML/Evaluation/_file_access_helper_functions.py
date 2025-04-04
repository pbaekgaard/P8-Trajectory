import pickle

def save_to_file(metadata, data):
    filename = metadata["filename"]
    with open(f"files/{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(metadata):
    filename = metadata["filename"]
    with open(f"files/{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    return data