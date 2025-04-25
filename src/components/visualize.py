import random
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib import rc, ticker


def visualize(evaluation_results: dict, only: List[str] = []) -> None:
    """
    Visualize Results!

    :param evaluation_results: Dict containing 'accuracy', 'compression_ratio', 'accuracy_individual_results', 'query_original_dataset_time', 'query_compressed_dataset_time', 'compression_time'
    :param only: Optional list of things to plot, allowed values: ['accuracy, compression_ratio, times']
    """
    # accuracy : float = evaluation_results["accuracy"]
    # compression_ratio : float = evaluation_results["compression_ratio"]
    # individual_accuracy_results : List[float] = evaluation_results["accuracy_individual_results"]
    # qorg_data_time : float = evaluation_results["query_original_dataset_time"]
    # qcomp_data_time : float = evaluation_results["query_compressed_dataset_time"]
    # compression_time : float = evaluation_results["compression_time"]
    only = [s.lower() for s in only]
    accuracy : float = 98.5
    compression_ratio : float = 2.4
    compression_ratios : List[float] = []
    accuracies : List[float] = []



    if("accuracy" in only or len(only) == 0):
        # MOCK DATA
        n = 7
        raw = [random.uniform(-2, 2) for _ in range(n)]
        offset = sum(raw) / n
        individual_accuracy_results : List[tuple[str, float]] = [(f"hello", round(accuracy - offset + r, 2)) for r in raw]
        results = [acc for _, acc in individual_accuracy_results]
        titles = ["Where", "Distance", "When", "How Long", "Count", "KNN", "Window"]
        # MOCK DATA END


        plt.bar(titles, results, color="skyblue")
        plt.axhline(accuracy, color="red", linestyle="solid", label=f"Overall Accuracy ({accuracy}%)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Individual Accuracy Results (Compression Ratio: {compression_ratio})")
        plt.legend()
        plt.tight_layout()
        plt.show()


    if("compression_ratio" in only or len(only) == 0):
        pass

    if("times" in only or len(only) == 0):
        # MOCK DATA
        qorg_data_time = 10.0
        qcomp_data_time = 3.5
        compression_time = 1200.2
        titles = ["Query Original Dataset Time", "Query Compressed Dataset Time", "Compression Time"]
        values = [qorg_data_time, qcomp_data_time, compression_time]
        # MOCK DATA END

        plt.figure(figsize=(8, 6))
        bars = plt.bar(titles, values, color="skyblue")
        plt.ylabel("Running Time (sec)")
        plt.yscale("log", base=2)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="black", fontsize=12, rotation=360, fontname="Comic Sans MS")
        plt.title("Query and Compression Times (log scale)")
        plt.tight_layout()
        plt.show()






if __name__ == "__main__":
    visualize({})
