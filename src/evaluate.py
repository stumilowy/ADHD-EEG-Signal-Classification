from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Control', 'ADHD']))

    # Rysowanie macierzy pomyłek
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'ADHD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Macierz Pomyłek (Confusion Matrix)")
    plt.show()