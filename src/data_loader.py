import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]

def load_and_split_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    return split_data(df, test_size=test_size, random_state=random_state)

def load_and_split_data_with_aggregation(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    return split_data_with_aggregation(df, test_size=test_size, random_state=random_state)

def split_data(df, test_size=0.2, random_state=42):
    X = df[channels]
    y = df['Class'].map({'Control': 0, 'ADHD': 1})
    groups = df['ID']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return  X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def split_data_with_aggregation(df, test_size=0.2, random_state=42):
    aggregations = {col: ['mean', 'std', 'var', 'min', 'max', 'median'] for col in channels}
    aggregations['Class'] = 'first'
    df_patients = df.groupby('ID').agg(aggregations)
    new_columns = []
    for col in df_patients.columns:
        if col[0] == 'Class':
            new_columns.append(col[0])
        else:
            new_columns.append(f'{col[0]}_{col[1]}')
    df_patients.columns = new_columns

    df_patients = df_patients.reset_index()

    X = df_patients.drop(columns=['ID', 'Class'])
    y = df_patients['Class'].map({'Control': 0, 'ADHD': 1})
    return train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )