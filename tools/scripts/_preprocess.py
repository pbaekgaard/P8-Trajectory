#!/usr/bin/env python3
"""
_preprocess.py
Contains functionality for preprocessing the downloaded data.
"""
import argparse
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Adds the current directory to sys.path

from _deduplicate import main as deduplicate
from _load_data import main as load_data
from _timestamporder import main as timestamporder
from _remove_illegal_points import main as remove_illegal_points
from _convert_timestamp_to_unix import main as timestamp_conversion
from _ten_trajectories import main as ten_trajectories
from _delimit_trajectory_length import main as delimit_trajectory_length

def parse_only(value):
    """
    Parse a comma-separated list enclosed in curly braces.
    For example, '--only={deduplication,timestamporder}' will return ['deduplication', 'timestamporder'].
    """
    # Extract everything between curly braces if present
    match = re.match(r'^\{(.*)\}$', value)
    if match:
        value = match.group(1)
    
    # Split by comma and strip whitespace
    steps = [s.strip() for s in value.split(",") if s.strip()]
    
    allowed = {'deduplication', 'timestamporder', 'limit_samplerate'}
    for step in steps:
        if step not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid preprocessing step: {step}. Allowed values are: {', '.join(sorted(allowed))}"
            )
    return steps

def format_step_list(steps):
    """
    Format a list of preprocessing steps with proper capitalization and conjunctions.
    For example, ['deduplication', 'timestamporder'] becomes 'Deduplication and Timestamp Ordering'.
    """
    # Dictionary for proper capitalization and word formatting
    step_names = {
        'deduplication': 'Deduplication',
        'timestamporder': 'Timestamp Ordering',
        'limit_samplerate': 'Limiting to avg. Sample Rate'
    }
    
    # Format each step
    formatted_steps = [step_names.get(step, step.capitalize()) for step in steps]
    
    # Join with commas and 'and' for the last element
    if len(formatted_steps) == 1:
        return formatted_steps[0]
    elif len(formatted_steps) == 2:
        return f"{formatted_steps[0]} and {formatted_steps[1]}"
    else:
        return f"{', '.join(formatted_steps[:-1])}, and {formatted_steps[-1]}"


def main(only=None):
    """
    Main function for preprocessing.
    
    :param only: Optional list of steps to run (e.g., ['deduplication', 'timestamporder', 'limit_samplerate']).
    """
    # Fix for when the 'only' parameter contains a single string with comma-separated values
    if only and isinstance(only, str) and ',' in only:
        only = [step.strip() for step in only.strip('{}').split(',')]



    # Convert to list if it's a single string
    if only and isinstance(only, str):
        only = [only]

    if only:
        # Dictionary for proper capitalization and word formatting
        step_names = {
            'deduplication': 'Deduplication',
            'timestamporder': 'Timestamp Ordering',
            'limit_samplerate': 'Limiting to avg. Sample Rate',
            'remove_illegal':  'Remove illegal coordinates.',
            'convert_timestamp_to_unix': 'Convert timestamp to unix.',
            'take_10_trajectories': 'taking 10 trajectories',
            'trajectory_length': 'trajectory length'
        }

        # Format each step
        formatted_steps = [step_names.get(step, step.capitalize()) for step in only]

        # Format output
        if len(formatted_steps) == 1:
            formatted_output = formatted_steps[0]
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 2:
            formatted_output = f"{formatted_steps[0]} and {formatted_steps[1]}"
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 3:
            formatted_output = f"{formatted_steps[0]} and {formatted_steps[1]} and {formatted_steps[2]}"
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 4:
            formatted_output = f"{formatted_steps[0]} and {formatted_steps[1]} and {formatted_steps[2]} and {formatted_steps[3]}"
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 5:
            formatted_output = f"{formatted_steps[0]} and {formatted_steps[1]} and {formatted_steps[2]} and {formatted_steps[3]} and {formatted_steps[4]}"
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 6:
            formatted_output = f"{formatted_steps[0]} and {formatted_steps[1]} and {formatted_steps[2]} and {formatted_steps[3]} and {formatted_steps[4]} and {formatted_steps[5]}"
            print(f"Preprocessing only the following steps: {formatted_output}")
        elif len(formatted_steps) == 7:
            print("Preprocessing all steps.")
        # Insert selective preprocessing logic here.
    else:
        print("Preprocessing all steps.")
        # Insert full preprocessing logic here.

    # Example processing work:
    print("Loading data...")
    if not only:
        only = [
        'deduplication',
        'remove_illegal',
        'limit_samplerate',
        'timestamporder',
        'convert_timestamp_to_unix',
        'take_10_trajectories',
        'trajectory_length:10'
        ]

    data = load_data()
    length_before = len(data)
    print(f"Processing Data...")
    for step in only:
        if step == "deduplication":
            print(f"Performing Deduplication...")
            data = deduplicate(data)
        elif step == "remove_illegal":
            print(f"Removing illegal coordinates...")
            data = remove_illegal_points(data)
        elif step == "timestamporder":
            print(f"Performing Timestamp Ordering...")
            data = timestamporder(data)
        elif step == "convert_timestamp_to_unix":
            print(f"Converting timestamp to unix...")
            data = timestamp_conversion(data)
        elif step == "take_10_trajectories":
            print("TAKING ONLY THE FIRST 10 TRAJECTORIESSS!!!!...")
            data = ten_trajectories(data)
        elif step.split(":")[0] == "trajectory_length":
            length = int(step.split(":")[1])
            print(f"Delimiting all trajectories to length {length}")
            data = delimit_trajectory_length(data, length)


    # ... your processing logic here ...
    print(f"Preprocessing complete! Removed {length_before - len(data)} entries.")
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess downloaded data.")
    parser.add_argument(
        "--only",
        type=parse_only,
        help="Specify which preprocessing steps to run, e.g. --only={deduplication,timestamporder}."
    )
    args = parser.parse_args()
    main(only=args.only)
