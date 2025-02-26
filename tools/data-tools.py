#!/usr/bin/env python3
"""
data_tools.py

A command-line utility that provides multiple actions:
    - download: Downloads and extracts the data using _get_data.py.
    - preprocess: Runs preprocessing using _preprocess.py.

You can run one or both actions in sequence. For example:
    data-tools download
    data-tools preprocess --only=deduplication,timestamporder
    data-tools download preprocess --only=deduplication,timestamporder
"""

import argparse
import sys

import scripts._get_data as _get_data
import scripts._preprocess as _preprocess


def parse_only(value):
    """
    Parse a comma-separated list enclosed in curly braces.
    For example, '--only=deduplication,timestamporder' will return ['deduplication', 'timestamporder'].
    """
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]  # Remove curly braces
    steps = [s.strip() for s in value.split(",") if s.strip()]
    allowed = {'deduplication', 'timestamporder', 'limit_samplerate'}
    for step in steps:
        if step not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid preprocessing step: {step}. Allowed values are: {', '.join(sorted(allowed))}"
            )
    return steps

def main():
    parser = argparse.ArgumentParser(
        prog="data-tools",
        description="A command-line tool for downloading and preprocessing data."
    )
    parser.add_argument(
        "actions",
        nargs="+",
        choices=["download", "preprocess"],
        help="Actions to execute in sequence. Choose one or both: download, preprocess"
    )
    parser.add_argument(
        "--only",
        type=parse_only,
        help="(Preprocess only) Specify which preprocessing steps to run, e.g. --only=deduplication,timestamporder."
    )

    args = parser.parse_args()

    # Execute each action in the order provided.
    for action in args.actions:
        if action == "download":
            print("Starting download process...")
            _get_data.main()  # Call the download function from _get_data.py
            print("Download process finished.\n")
        elif action == "preprocess":
            print("Starting preprocessing...")
            _preprocess.main(only=args.only)  # Pass the parsed '--only' list (or None)
            print("Preprocessing finished.\n")
        else:
            # This branch should not occur because of argparse's choices.
            print(f"Unknown action: {action}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
