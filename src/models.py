from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_pipeline(model_type="random_forest"):
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Nieznany typ modelu")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])