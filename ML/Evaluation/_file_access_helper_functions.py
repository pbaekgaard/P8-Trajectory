import pickle
import os

def save_to_file(metadata, data):
    filename = metadata["filename"]
    version = metadata["version"]
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version}-{filename}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(metadata):
    filename = metadata["filename"]
    if "version" in metadata:
        version = metadata["version"]
    else:
        version = find_newest_version()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version}-{filename}.pkl"), "rb") as f:
        data = pickle.load(f)
    return data


def find_newest_version():
    Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
    files = [f for f in os.listdir(Dir) if os.path.isfile(os.path.join(Dir, f)) and f.endswith('.pkl')]
    version_number = 0
    for file in files:
        split_file = file.split('-')
        if len(split_file) > 1:
            version_number = max(int(split_file[0]), version_number)

    return version_number



