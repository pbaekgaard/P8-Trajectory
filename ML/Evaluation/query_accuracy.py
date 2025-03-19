from tools.scripts._load_data import main as _load_data, load_compressed_data as _load_compressed_data
from datetime import datetime
import pandas as pd

def query_accuracy_evaluation(queries):
    compressed_data, original_dataset = _load_compressed_data(), _load_data()
    query_original_dataset(original_dataset, queries)

def query_original_dataset(dataset, queries):
    print(dataset)
    where_query = queries["where"]
    group_by_df = dataset.groupby("trajectory_id")

    where_query_results_df = pd.DataFrame([], columns=["trajectory_id", "less_than", "greater_than"])

    for trajectory_id, group_df in group_by_df:
        less_than = group_df[group_df["timestamp"] <= where_query]["timestamp"]
        if less_than.empty: continue
        first_less_than = less_than.iloc[-1]

        greater_than = group_df[group_df["timestamp"] >= where_query]["timestamp"]
        first_greater_than = greater_than.iloc[-1] if not greater_than.empty else None
        where_query_results_df.loc[len(where_query_results_df)] = [trajectory_id, first_less_than, first_greater_than]
    print(where_query_results_df)

def query_compressed_data():
    pass

def create_queries():
    return {
        "where": "2008-02-02 15:38:08"
    }

if __name__ == '__main__':
    query_accuracy_evaluation(create_queries())