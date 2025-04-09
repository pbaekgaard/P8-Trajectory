from ML.Evaluation._file_access_helper_functions import save_to_file, load_data_from_file


def query_accuracy_evaluation():
    # Load compressed results and original results and calculate accuracy
    # compressed_results = load_data_from_file("compressed_query_results")
    original_results = load_data_from_file({
        "filename": "test-original_query_results"
    })
    queries = load_data_from_file({
        "filename": "queries_for_evaluation",
    })


    # print(compressed_results)
    print(original_results)


query_accuracy_evaluation()

