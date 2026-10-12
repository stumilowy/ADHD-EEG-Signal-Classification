import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_and_split_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)

    channels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
    ]

    X = df[channels]
    y = df['Class'].map({'Control': 0, 'ADHD': 1})
    groups = df['ID']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return  X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]