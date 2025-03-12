from embed import *
import tools.scripts._get_data as _get_data
import tools.scripts._load_data as _load_data



if __name__ == '__main__':
    _get_data.main()
    df = _load_data.main()

    print(df)