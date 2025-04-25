import pandas as pd

def main(data: pd.DataFrame) -> pd.DataFrame:
    data['timestamp'] = pd.to_datetime(data['timestamp']).astype('int64') // 10**9 # .view('int64') gives the timestamp in nanoseconds which is 10^9 seconds
    return data