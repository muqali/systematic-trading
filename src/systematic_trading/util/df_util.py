import pandas as pd

def show_dataframe_diff(df1: pd.DataFrame, df2: pd.DataFrame, max_rows: int = 20):
    print("shape:")
    print("df1:", df1.shape)
    print("df2:", df2.shape)

    if not df1.index.equals(df2.index):
        print("\nindex differs")
        only_in_df1 = df1.index.difference(df2.index)
        only_in_df2 = df2.index.difference(df1.index)
        print("only in df1:", list(only_in_df1[:max_rows]))
        print("only in df2:", list(only_in_df2[:max_rows]))

    if not df1.columns.equals(df2.columns):
        print("\ncolumns differ")
        only_in_df1 = df1.columns.difference(df2.columns)
        only_in_df2 = df2.columns.difference(df1.columns)
        print("only in df1:", list(only_in_df1))
        print("only in df2:", list(only_in_df2))

    common_index = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)

    a = df1.loc[common_index, common_columns]
    b = df2.loc[common_index, common_columns]

    try:
        diff = a.compare(b)
        if diff.empty:
            print("\nno value differences on common index/columns")
        else:
            print("\nvalue differences:")
            print(diff.head(max_rows))
    except Exception as e:
        print("\ncould not compare values:", e)
