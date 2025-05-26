import os
import random
import sys
import ast
from typing import List, Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
        plt.axhline(accuracy, color="red", linestyle="solid", label=f"Overall Accuracy ({accuracy}%)")
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.title(f"Individual Accuracy Results (Compression Ratio: {compression_ratio})", fontweight="bold")
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360, fontname="Comic Sans MS")

        plt.legend(loc='upper left', bbox_to_anchor=(0.535, 0.85), framealpha=0.5)
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
        values = [qorg_data_time, ml_time, MRT_time, OSTC_time, qcomp_data_time]

        total_time = sum(values)


        titles = ["Query Original Dataset", "Reference set construction", "MRT Search", "OSTC", "Query Compressed Dataset"]
        combined_labels = [
            f"{val:.2f} \n ({(val / total_time) * 100:.1f}%)" for val in values
        ]

        percentage_labels = [
            f"{(val / total_time) * 100:.2f}%" for val in values
        ]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(titles, values, color="skyblue")
        plt.ylabel("Running Time (sec)", fontsize=12)
        plt.yscale("log", base=2)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.xticks(fontsize=12, rotation=15, ha='center')  # Rotate x-axis labels
        ax.bar_label(bars, labels=combined_labels, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360, fontname="Comic Sans MS")
        plt.title("Query and Compression Times (log scale)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig("times.svg", format='svg')
        plt.show()

def visualize_comp_acc(evaluation_results_len_50, evaluation_results_len_250):
    accuracy_len_50: float = round(evaluation_results_len_50["accuracy"] * 100, 2)
    compression_ratio_len_50: float = round(evaluation_results_len_50["compression_ratio"], 2)

    accuracy_len_250: float = round(evaluation_results_len_250["accuracy"] * 100, 2)
    compression_ratio_len_250: float = round(evaluation_results_len_250["compression_ratio"], 2)


    # MOCK DATA
    # n = 7
    # raw = [random.uniform(-2, 2) for _ in range(n)]
    # offset = sum(raw) / n
    # individual_accuracy_results : List[tuple[str, float]] = [(f"hello", round(accuracy - offset + r, 2)) for r in raw]
    # results = [acc for _, acc in individual_accuracy_results]
    # titles = ["Where", "Distance", "When", "How Long", "Count", "KNN", "Window"]
    # MOCK DATA END

    titles = ["Length 50", "Length 250"]
    compression_ratios = [compression_ratio_len_50, compression_ratio_len_250]
    accuracies = [accuracy_len_50, accuracy_len_250]

    x = np.arange(len(titles))
    width = 0.35

    fig, ax1 = plt.subplots()

    bars1 = ax1.bar(x - width / 2, compression_ratios, width, color='skyblue', label='Compression Ratio')
    ax1.set_ylabel('Compression Ratio', color='skyblue')

    ax1.set_ylim(0, 2)
    ax1.bar_label(bars1, fmt='%.2f', label_type='center', padding=3)

    ax2 = ax1.twinx()

    bars2 = ax2.bar(x + width / 2, accuracies, width, color='mediumseagreen', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='mediumseagreen')

    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt='%.2f%%', label_type='center', padding=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(titles)

    plt.tight_layout()
    plt.savefig("acc-comp-comparisons.svg", format='svg')
    plt.show()

def visualize_mrt_huge(evaluation_results: dict) -> None:
    MRT_time_len_50: float = evaluation_results["MRT_time_len_50"]
    MRT_time_len_250: float = evaluation_results["MRT_time_len_250"]

    titles = ["MRT Search Length 50", "MRT Search Length 250"]
    values = [MRT_time_len_50, MRT_time_len_250]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(titles, values, color="skyblue")
    plt.ylabel("Running Time (sec)", fontsize=12)
    plt.yscale("log", base=2)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.bar_label(bars, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360,
                 fontname="Comic Sans MS")
    plt.title("Query and Compression Times (log scale)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mrt-huge.svg", format='svg')
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
    org_query_res_len_50 : dict = load_data_from_file({
        "filename": "original_query_results",
        "version": 1
    })


    clustering_method_len_50, clustering_param_len_50, batch_size_len_50, d_model_len_50, num_heads_len_50, clustering_metric_len_50, num_layers_len_50, compression_ratio_len_50, ml_time_len_50, compression_time_len_50, Total_MRT_time_len_50, Total_OSTC_time_len_50, querying_time_len_50, total_time_len_50, accuracy_individual_results_len_50, score_len_50 = get_best_params()
    evaluation_results_len_50 = {}

    evaluation_results_len_50["query_original_dataset_time"] = org_query_res_len_50['times']['querying_time']
    evaluation_results_len_50["query_compressed_dataset_time"] = querying_time_len_50 / 10**9
    evaluation_results_len_50["MRT_time"] = Total_MRT_time_len_50 / 10**3
    evaluation_results_len_50["OSTC_time"] = Total_OSTC_time_len_50 / 10**3
    evaluation_results_len_50["ml_time"] = ml_time_len_50 / 10**9
    evaluation_results_len_50["accuracy"] = score_len_50
    evaluation_results_len_50["compression_ratio"] = compression_ratio_len_50
    evaluation_results_len_50["accuracy_individual_results"] = ast.literal_eval(accuracy_individual_results_len_50)

    org_query_res_len_250: dict = load_data_from_file({
        "filename": "original_query_results",
        "version": 2
    })

    compressed_query_res_len_250 : dict = load_data_from_file({
        "filename": "compressed_query_results",
        "version": 2
    })
    evaluation_results_len_250 : dict = load_data_from_file({
        "filename": "evaluation",
        "version": 2
    })

    evaluation_results_len_250['query_original_dataset_time'] = org_query_res_len_250['times']['querying_time'] / 10**9
    evaluation_results_len_250['query_compressed_dataset_time'] = compressed_query_res_len_250['times']['querying_time'] / 10**9
    evaluation_results_len_250["MRT_time"] = compressed_query_res_len_250['times']["Total_MRT_time"] / 10**3
    evaluation_results_len_250["OSTC_time"] = compressed_query_res_len_250['times']["Total_OSTC_time"] / 10**3
    evaluation_results_len_250['ml_time'] = compressed_query_res_len_250['times']['ml_time']  / 10**9

    mrt_huge_results = {}

    mrt_huge_results["MRT_time_len_50"] = Total_MRT_time_len_50 / 10**3
    mrt_huge_results["MRT_time_len_250"] = compressed_query_res_len_250['times']["Total_MRT_time"]  / 10**3

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

    visualize(evaluation_results_len_50)
    # visualize_trajectories(visualize_traj_dict)
    visualize_mrt_huge(mrt_huge_results)
    visualize_comp_acc(evaluation_results_len_50, evaluation_results_len_250)
