import sys
from pympler import asizeof

def compression_ratio(dataset):
    size_of_dataset = sys.getsizeof(dataset)

    compressed_data, merged_df = compress(dataset)
    size_of_compressed_data = (asizeof.asizeof(compressed_data) + \
                merged_df.drop(columns=['timestamp_corrected']).memory_usage(deep=True).sum() + \
                merged_df['timestamp_corrected'].apply(lambda x: asizeof.asizeof(x) if isinstance(x, dict) and x else 0).sum())

    compression_ratio = size_of_dataset / size_of_compressed_data

    return compression_ratio

def compress(dataset):
    return dataset
