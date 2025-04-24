import pandas as pd
def main(data, length):
    trajectories = []
    for _, group in data.groupby('trajectory_id'):
        trajectories.append(group.head(length))
    return pd.concat(trajectories, ignore_index=True)
