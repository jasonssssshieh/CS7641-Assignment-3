from sklearn import  datasets, metrics
import abalone_EMTestCluster as emtc
import pandas as pd

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
    abalone_data = pd.read_csv("abalone_data.csv")
    dft, mapping = encode_target(abalone_data, "rings")
    X = (dft.ix[:, :-1])
    y = dft.ix[:, -1]

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,41), plot=True, targetcluster=5, stats=True)
    tester.run()

