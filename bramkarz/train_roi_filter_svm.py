import os
import cv2
import numpy as np
import glob
import random
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def load_bramkarz_data(folder_1, folder_0, multiplier=3):
    """
    Wczytuje wszystkie znaki (klasa 1) i losuje proporcjonalną ilość śmieci (klasa 0).
    """
    hog = get_hog_descriptor()
    X_features = []
    y_labels = []

    # 1. Wczytywanie ZNAKÓW (Klasa 1)
    print(f"Wczytywanie pozytywów (Znaków) z: {folder_1}...")
    paths_1 = glob.glob(os.path.join(folder_1, '*.png'))

    for img_path in paths_1:
        img = cv2.imread(img_path)
        if img is None: continue

        # Upewniamy się, że rozmiar to 64x64 przed HOG-iem
        img_resized = cv2.resize(img, (64, 64))
        hog_vector = hog.compute(img_resized).flatten()

        X_features.append(hog_vector)
        y_labels.append(1)  # 1 oznacza ZNAK

    num_signs = len(paths_1)
    print(f"Załadowano {num_signs} znaków drogowych.")

    # 2. Wczytywanie ŚMIECI (Klasa 0) - SMART SAMPLING
    print(f"Wczytywanie negatywów (Śmieci) z: {folder_0}...")
    paths_0 = glob.glob(os.path.join(folder_0, '*.png'))

    # Obliczamy ile śmieci chcemy wziąć
    target_trash_count = min(len(paths_0), num_signs * multiplier)
    print(f"Znaleziono {len(paths_0)} zdjęć tła. Losuję z nich {target_trash_count} sztuk do treningu...")

    sampled_paths_0 = random.sample(paths_0, target_trash_count)

    for img_path in sampled_paths_0:
        img = cv2.imread(img_path)
        if img is None: continue

        img_resized = cv2.resize(img, (64, 64))
        hog_vector = hog.compute(img_resized).flatten()

        X_features.append(hog_vector)
        y_labels.append(0)  # 0 oznacza ŚMIEĆ/TŁO

    return np.array(X_features), np.array(y_labels)


if __name__ == "__main__":
    # ==========================================
    # KONFIGURACJA ŚCIEŻEK
    # ==========================================
    FOLDER_ZNAKOW = 'train/1'
    FOLDER_SMIECI = 'train/0'
    MODEL_OUTPUT = 'bramkarz_model.pkl'

    # ==========================================
    # ETAP 1: Przygotowanie Danych
    # ==========================================
    X, y = load_bramkarz_data(FOLDER_ZNAKOW, FOLDER_SMIECI, multiplier=3)
    print(f"\nŁączny zbiór danych: {X.shape[0]} wektorów HOG.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # ETAP 2: Inicjalizacja Potoku (Pipeline)
    # ==========================================
    print("\nInicjalizacja potoku: StandardScaler + SVM...")

    # Tworzymy pipeline. Najpierw dane zostaną znormalizowane, a potem trafią do SVM.
    # Ustawiamy podstawowe parametry SVM, które są stałe.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    # ==========================================
    # ETAP 3: Poszukiwanie Najlepszych Parametrów
    # ==========================================
    print("Rozpoczynam Grid Search (kros-walidacja 5-krotna)... To może potrwać!")

    # Definiujemy siatkę parametrów do przetestowania.
    # Przedrostek 'svm__' wskazuje, że parametr dotyczy etapu 'svm' w naszym pipeline.
    param_grid = {
        'svm__kernel': ['linear', 'rbf'],  # Testujemy zarówno szybkie jądro liniowe, jak i dokładniejsze RBF
        'svm__C': [0.1, 1.0, 10.0]  # Testujemy różną siłę kary za błędy
    }

    # n_jobs=-1 angażuje wszystkie rdzenie procesora do równoległych obliczeń
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\nZakończono Grid Search! Najlepsze parametry to: {grid_search.best_params_}")

    # Pobieramy najlepszy model z całego przeszukiwania
    best_model = grid_search.best_estimator_

    # ==========================================
    # ETAP 4: Ewaluacja Najlepszego Modelu
    # ==========================================
    print("\nSprawdzanie skuteczności najlepszego modelu na danych testowych (20%)...")
    y_pred = best_model.predict(X_test)

    print(f"Dokładność ogólna: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Szczegółowy raport:")
    print(classification_report(y_test, y_pred, target_names=["Tło/Śmieć (0)", "Znak (1)"]))

    # ==========================================
    # ETAP 5: Zapis Modelu
    # ==========================================
    # Używamy joblib, co rozwiązuje wcześniejsze problemy w pliku GUI
    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"\nZapisano wytrenowanego Bramkarza (wraz ze skalerem) do pliku: {MODEL_OUTPUT}")