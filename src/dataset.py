import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/adhdata.csv")

channels = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz'
]

X = df[channels]
y = df['Class'].map({'Control': 0, 'ADHD': 1})
groups = df['ID']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 4. Użycie Pipeline (skalowanie danych + model)
# Choć Las Losowy radzi sobie bez skalowania, to dobra praktyka inżynierska
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
])

# 5. Trening i predykcja
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 6. Ewaluacja (dokładniejsze metryki zamiast samego accuracy)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Control', 'ADHD']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

