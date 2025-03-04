#!/usr/bin/env python3
import os

from components.loaddata import main as load_data
from components.mapmatching import mapmatch


def main():
    data = load_data(dataType='parquet')



if __name__ == "__main__":
    main()
