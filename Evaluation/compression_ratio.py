import sys
from pympler import asizeof

def compression_ratio(dataset, compressed_dataset, merged_df):
    size_of_dataset = dataset.memory_usage(deep=True).sum()

    
    size_of_compressed_data = (asizeof.asizeof(compressed_dataset) + \
                merged_df.drop(columns=['timestamp_corrected']).memory_usage(deep=True).sum() + \
                merged_df['timestamp_corrected'].apply(lambda x: asizeof.asizeof(x) if isinstance(x, dict) and x else 0).sum())

    compression_ratio = size_of_dataset / size_of_compressed_data

    return compression_ratio

