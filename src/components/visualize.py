import os
import random
import sys
import ast
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc, ticker

sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../../")))

from Evaluation._file_access_helper_functions import load_data_from_file, get_best_params
from tools.scripts._preprocess import main as _load_data


def visualize(evaluation_results: dict, only: List[str] = []) -> None:
    """
    Visualize Results!

    :param evaluation_results: Dict containing 'accuracy', 'compression_ratio', 'accuracy_individual_results', 'query_original_dataset_time', 'query_compressed_dataset_time', 'compression_time'
    :param only: Optional list of things to plot, allowed values: ['accuracy, compression_ratio, times']
    """
    accuracy : float = round(evaluation_results["accuracy"] * 100, 2)
    compression_ratio : float = round(evaluation_results["compression_ratio"], 2)
    individual_accuracy_results : List[float] = evaluation_results["accuracy_individual_results"]
    qorg_data_time : float = evaluation_results["query_original_dataset_time"]
    qcomp_data_time : float = evaluation_results["query_compressed_dataset_time"]
    MRT_time : float = evaluation_results["MRT_time"]
    OSTC_time : float = evaluation_results["OSTC_time"]
    ml_time : float = evaluation_results["ml_time"]
    only = [s.lower() for s in only]
    # accuracy : float = 98.5
    # compression_ratio : float = 2.4
    compression_ratios : List[float] = []
    accuracies : List[float] = []



    if "accuracy" in only or len(only) == 0:
        # MOCK DATA
        # n = 7
        # raw = [random.uniform(-2, 2) for _ in range(n)]
        # offset = sum(raw) / n
        # individual_accuracy_results : List[tuple[str, float]] = [(f"hello", round(accuracy - offset + r, 2)) for r in raw]
        # results = [acc for _, acc in individual_accuracy_results]
        # titles = ["Where", "Distance", "When", "How Long", "Count", "KNN", "Window"]
        # MOCK DATA END

        titles = [title for title, _ in individual_accuracy_results]
        results = [acc * 100 for _, acc in individual_accuracy_results]

        bars = plt.bar(titles, results, color="skyblue")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.bar(titles, results, color="skyblue")
        plt.axhline(accuracy, color="red", linestyle="solid", label=f"Overall Accuracy ({accuracy}%)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Individual Accuracy Results (Compression Ratio: {compression_ratio})")
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360, fontname="Comic Sans MS")

        plt.legend()
        plt.tight_layout()
        plt.savefig("accuracy.svg", format='svg')
        plt.show()


    if "compression_ratio" in only or len(only) == 0:
        pass

    if "times" in only or len(only) == 0:
        # MOCK DATA
        # qorg_data_time = 10.0
        # qcomp_data_time = 3.5
        # compression_time = 1200.2
        # MOCK DATA END

        titles = ["Query Original Dataset Time", "Query Compressed Dataset Time", "MRT Search Time", "OSTC Time", "Reference set construction time"]
        values = [qorg_data_time, qcomp_data_time, MRT_time, OSTC_time, ml_time]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(titles, values, color="skyblue")
        plt.ylabel("Running Time (sec)")
        plt.yscale("log", base=2)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.xticks(rotation=10, ha='center')  # Rotate x-axis labels
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360, fontname="Comic Sans MS")
        plt.title("Query and Compression Times (log scale)")
        plt.tight_layout()
        plt.savefig("times.svg", format='svg')
        plt.show()


def visualize_trajectories(visualize_dict: dict, only: List[str] = []) -> None:
    compressed_dataset = visualize_dict["compressed"]
    merged_df = visualize_dict["reference_set"]
    original_dataset = visualize_dict["original"]

    random_compressed_traj_key = random.choice(list(compressed_dataset))
    random_compressed_traj_value = compressed_dataset[random_compressed_traj_key]
    last_point = None

    plt.figure(figsize=(8, 6))
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    color_of_original = "black"

    original_trajectory_to_plot = original_dataset[original_dataset["trajectory_id"] == random_compressed_traj_key]
    if "not_rest" in only or len(only) == 0:
        for triple in random_compressed_traj_value:
            ref_traj_id = triple[0]
            start_index = triple[1]
            end_index = triple[2]

            trajectory_to_plot = merged_df[merged_df["trajectory_id"] == ref_traj_id].iloc[start_index:end_index + 1]
            if last_point is not None:
                list_to_plot = pd.concat([last_point, trajectory_to_plot.iloc[[0]]])
                plt.plot(list_to_plot["longitude"], list_to_plot["latitude"], color='grey')

            if ref_traj_id == random_compressed_traj_key:
                plt.plot(trajectory_to_plot["longitude"], trajectory_to_plot["latitude"], color=color_of_original)
            else:
                plt.plot(trajectory_to_plot["longitude"], trajectory_to_plot["latitude"])

            last_point = trajectory_to_plot.iloc[[-1]]

        plt.plot(original_trajectory_to_plot["longitude"], original_trajectory_to_plot["latitude"], color=color_of_original)
        plt.show()

    if "rest" in only or len(only) == 0:
        for triple in random_compressed_traj_value:
            ref_traj_id = triple[0]
            start_index = triple[1]
            end_index = triple[2]

            if ref_traj_id == random_compressed_traj_key: continue

            trajectory_to_plot = merged_df[merged_df["trajectory_id"] == ref_traj_id].iloc[start_index:end_index + 1]

            plt.plot(trajectory_to_plot["longitude"], trajectory_to_plot["latitude"])

        plt.plot(original_trajectory_to_plot["longitude"], original_trajectory_to_plot["latitude"], color=color_of_original)
        plt.show()


if __name__ == "__main__":
    org_query_res : dict = load_data_from_file({
        "filename": "original_query_results",
        "version": 1
    })

    clustering_method, clustering_param, batch_size, d_model, num_heads, clustering_metric, num_layers, compression_ratio, ml_time, compression_time, Total_MRT_time, Total_OSTC_time, querying_time, total_time, accuracy_individual_results, score = get_best_params()
    evaluation_results = {}

    evaluation_results["query_original_dataset_time"] = org_query_res['times']['querying_time']
    evaluation_results["query_compressed_dataset_time"] = querying_time / 10**9
    evaluation_results["MRT_time"] = Total_MRT_time / 10**3
    evaluation_results["OSTC_time"] = Total_OSTC_time / 10**3
    evaluation_results["ml_time"] = ml_time / 10**9
    evaluation_results["accuracy"] = score
    evaluation_results["compression_ratio"] = compression_ratio
    evaluation_results["accuracy_individual_results"] = ast.literal_eval(accuracy_individual_results)

    # compressed_query_res : dict = load_data_from_file({
    #     "filename": "compressed_query_results",
    #     "version": 3
    # })
    # evaluation_results : dict = load_data_from_file({
    #     "filename": "evaluation",
    #     "version": 3
    # })
    # evaluation_results['query_original_dataset_time'] = org_query_res['times']['querying_time']
    # evaluation_results['query_compressed_dataset_time'] = compressed_query_res['times']['querying_time']
    # evaluation_results['compression_time'] = compressed_query_res['times']['compression_time']
    # evaluation_results['ml_time'] = compressed_query_res['times']['ml_time']
    #
    # visualize_traj_dict = {}
    #
    # visualize_traj_dict["compressed"] = compressed_query_res["compressed_dataset"]
    # visualize_traj_dict["reference_set"] = compressed_query_res["merged_dataset"]
    # visualize_traj_dict["original"] = _load_data()

    # MOCK DATA
#     visualize_traj_dict["original"] = pd.DataFrame([
#     [0, 1201956968, 116.51172, 39.92123],  # Trajectory 1
#     [0, 1201958410, 116.51222, 39.92173],
#     [0, 1201965600, 116.51372, 39.92323],
#
#     [1, 1201951200, 116.50000, 39.90000],  # Trajectory 2
#     [1, 1201952100, 116.51000, 39.91000],
#
#     [2, 1201966200, 116.55000, 39.95000],  # Trajectory 3
#     [2, 1201966320, 116.55200, 39.95200],
#
#     [3, 1201949400, 116.50050, 39.91050],  # Trajectory 4
#     [3, 1201950300, 116.52050, 39.93050],
#     [3, 1201951200, 116.54050, 39.95050],
#
#     [4, 1201969800, 116.57000, 39.97000],  # Trajectory 5
#     [4, 1201970100, 116.58000, 39.98000],
#
#     [5, 1201972800, 116.59000, 39.99000],  # Trajectory 6
#     [5, 1201973100, 116.60000, 39.99200],
#     [5, 1201973400, 116.61000, 39.99300]
# ], columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    visualize(evaluation_results)
    # visualize_trajectories(visualize_traj_dict)
