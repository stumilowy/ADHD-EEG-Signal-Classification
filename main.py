from src import load_and_split_data
from src import split_data
from src import get_pipeline
from src import evaluate_model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    import pandas as pd

    # 1. Wczytanie surowych danych
    df = pd.read_csv("data/adhdata.csv")

    # 2. Definicja kanałów
    channels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
    ]

    # 3. Definiujemy, jakie statystyki chcemy wyciągnąć z przebiegu całego badania dla każdego kanału
    # Dla każdego z 19 kanałów policzymy te 6 wartości:
    aggregations = {col: ['mean', 'std', 'var', 'min', 'max', 'median'] for col in channels}

    # Dodajemy informację o klasie pacjenta ('first' bierze pierwszą napotkaną klasę, bo w obrębie pacjenta się ona nie zmienia)
    aggregations['Class'] = 'first'

    print("Rozpoczynam kompresję 2 milionów wierszy. To potrwa kilka sekund...")

    # 4. Magia Pandas: Grupujemy wszystkie skany po ID pacjenta i wyliczamy statystyki
    df_patients = df.groupby('ID').agg(aggregations)

    # 5. Spłaszczanie nazw kolumn
    # Pandas stworzy nazwy dwupoziomowe (np. kanał Fp1, pod nim 'mean'). Chcemy z tego zrobić 'Fp1_mean'
    nowe_kolumny = []
    for col in df_patients.columns:
        if col[0] == 'Class':
            nowe_kolumny.append('Class')
        else:
            nowe_kolumny.append(f"{col[0]}_{col[1]}")  # np. Fp1_mean, Cz_std

    df_patients.columns = nowe_kolumny

    # Resetujemy indeks, żeby 'ID' znowu stało się zwykłą kolumną w tabeli
    df_patients = df_patients.reset_index()

    # Koniec! Możesz zobaczyć swoją nową tabelę:
    print(f"Nowy rozmiar tabeli: {df_patients.shape}")  # Zobaczysz (121, 116)
    print(df_patients.head())

    # Opcjonalnie: możesz zapisać sobie tę małą tabelkę, żeby nie musieć już wczytywać tych 2 milionów wierszy!
    # df_patients.to_csv("../data/adhdata_patients.csv", index=False)

    # 1. Zdefiniowanie X: Bierzemy wszystkie 114 wyliczonych statystyk (cech),
    # po prostu odrzucając z tabeli kolumny, które nie są cechami sygnału.
    X = df_patients.drop(columns=['ID', 'Class'])

    # 2. Zdefiniowanie y: Zamiana etykiet na numeryczne 0 i 1
    y = df_patients['Class'].map({'Control': 0, 'ADHD': 1})

    # 3. Podział danych.
    # Każdy wiersz to teraz 1 pacjent, więc zwykły train_test_split
    # naturalnie spełnia wymóg "Subject Split" (nie miesza danych tej samej osoby).
    # Opcja stratify=y gwarantuje, że zachowamy proporcje klas.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )
    #
    # print("Wczytywanie i dzielenie danych...")
    # X_train, X_test, y_train, y_test = split_data(df_patients)

    print("Inicjalizacja i trening modelu...")
    rfPipeline = get_pipeline("random_forest")
    gbPipeline = get_pipeline("gradient_boosting")
    #rfPipeline.fit(X_train, y_train)
    gbPipeline.fit(X_train, y_train)

    print("Ewaluacja modelu...")
    print(len(X_test))
    print(len(y_test))
    #evaluate_model(rfPipeline, X_test, y_test)
    evaluate_model(gbPipeline, X_test, y_test)