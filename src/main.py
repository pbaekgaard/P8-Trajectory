import os

from components.loaddata import main as load_data


def main():
    data = load_data(dataType='parquet')
    print(data)

if __name__ == "__main__":
    main()
