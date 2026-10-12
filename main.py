from src import load_and_split_data
from src import get_pipeline
from src import evaluate_model

if __name__ == "__main__":
    print("Wczytywanie i dzielenie danych...")
    X_train, X_test, y_train, y_test = load_and_split_data("data/adhdata.csv")

    print("Inicjalizacja i trening modelu...")
    pipeline = get_pipeline("random_forest")
    pipeline.fit(X_train, y_train)

    print("Ewaluacja modelu...")
    evaluate_model(pipeline, X_test, y_test)