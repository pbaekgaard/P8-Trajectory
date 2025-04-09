from tools.scripts._load_data import main as _load_data
import sys
import timeit
import torch

def compression_ratio():
    dataset = _load_data()
    size_of_dataset = sys.getsizeof(dataset)

    # compression_time = timeit.timeit(compress, number=10) / 10
    compression_time = None
    compressed_data = compress(dataset)

    size_of_compressed_data = sys.getsizeof(compressed_data)
    compression_ratio = size_of_dataset / size_of_compressed_data

    return compression_ratio, compression_time


def compress(dataset):
    with torch.no_grad():
        for i in range(0, 100000):
            continue
    return dataset