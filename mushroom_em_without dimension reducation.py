from sklearn import  datasets, metrics
import mushroom_EMTestCluster as emtc
import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

if __name__ == "__main__":
    mushroom_data = pd.read_csv("mushroom_data.csv")
    dft, mapping = encode_target(mushroom_data, "class")
    X = (dft.ix[:, :-1])
    y = dft.ix[:, -1]
    X = StandardScaler().fit_transform(X)
    
    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,35), plot=True, targetcluster=2, stats=True)
    tester.run()


