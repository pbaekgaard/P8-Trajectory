import sys

def compression_ratio(dataset):
    size_of_dataset = sys.getsizeof(dataset)

    compressed_data = compress(dataset)
    size_of_compressed_data = sys.getsizeof(compressed_data)

    compression_ratio = size_of_dataset / size_of_compressed_data

    return compression_ratio

def compress(dataset):
    return dataset
