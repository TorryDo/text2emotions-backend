import pandas as pd


def try_rm_dupl_df(df: pd.DataFrame):
    index = df[df.duplicated() == True].index
    df.drop(index, axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # removing duplicated text
    index = df[df['Text'].duplicated() == True].index
    df.drop(index, axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
