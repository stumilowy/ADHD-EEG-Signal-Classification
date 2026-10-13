from src import get_pipeline
from src import evaluate_model
from src import load_and_split_data_with_aggregation
from src import load_and_split_data

if __name__ == "__main__":

    print("Inicjalizacja modeli...")
    rf_pipeline_agg = get_pipeline("random_forest")
    rf_pipeline = get_pipeline("random_forest")
    gb_pipeline_agg = get_pipeline("gradient_boosting")
    gb_pipeline = get_pipeline("gradient_boosting")

    print("Wczytywanie i dzielenie danych agregowanych...")
    X_train_agg, X_test_agg, y_train_agg, y_test_agg = load_and_split_data_with_aggregation('data/adhdata.csv')
    X_train, X_test, y_train, y_test = load_and_split_data('data/adhdata.csv')

    print("Trening modeli...")
    rf_pipeline_agg.fit(X_train_agg, y_train_agg)
    rf_pipeline.fit(X_train, y_train)

    print("Ewaluacja modeli...")
    evaluate_model(rf_pipeline_agg, X_test_agg, y_test_agg, "Random Forest (agg)")
    evaluate_model(rf_pipeline_agg, X_test_agg, y_test_agg, " Gradient Boosting (agg)")

    evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    evaluate_model(rf_pipeline, X_test, y_test, "Gradient Boosting")

