"""
utils.py — Moduł z funkcjami współdzielonymi przez cały projekt.

Zawiera konfigurację HOG, mapowanie klas GTSRB oraz funkcje pomocnicze
wykorzystywane w treningu, ewaluacji i GUI.
"""

import cv2
import numpy as np
import os
import glob

# ============================================================
# KONFIGURACJA STAŁYCH
# ============================================================

# Mapowanie ID klas GTSRB → czytelne nazwy znaków drogowych
CLASS_LABELS = {
    2:  "Ograniczenie prędkości (50 km/h)",
    12: "Droga z pierwszeństwem",
    14: "Znak STOP",
    17: "Zakaz wjazdu",
    35: "Nakaz jazdy prosto",
}

# Lista ID wybranych klas (do filtrowania z bazy GTSRB)
TARGET_CLASSES = [2, 12, 14, 17, 35]

# Docelowy rozmiar obrazu po przeskalowaniu
TARGET_SIZE = (64, 64)

# Domyślne parametry deskryptora HOG
HOG_WIN_SIZE = (64, 64)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9


# ============================================================
# FUNKCJE WSPÓŁDZIELONE
# ============================================================

def get_hog_descriptor(win_size=None, block_size=None, block_stride=None,
                       cell_size=None, nbins=None):
    """
    Tworzy i zwraca obiekt cv2.HOGDescriptor z konfigurowalnymi parametrami.

    Parameters
    ----------
    win_size : tuple of int, optional
        Rozmiar okna detekcji (szerokość, wysokość). Domyślnie (64, 64).
    block_size : tuple of int, optional
        Rozmiar bloku normalizacji. Domyślnie (16, 16).
    block_stride : tuple of int, optional
        Krok przesuwania bloku. Domyślnie (8, 8).
    cell_size : tuple of int, optional
        Rozmiar komórki histogramu. Domyślnie (8, 8).
    nbins : int, optional
        Liczba binów histogramu orientacji gradientów. Domyślnie 9.

    Returns
    -------
    cv2.HOGDescriptor
        Skonfigurowany deskryptor HOG gotowy do użycia.
    """
    return cv2.HOGDescriptor(
        win_size or HOG_WIN_SIZE,
        block_size or HOG_BLOCK_SIZE,
        block_stride or HOG_BLOCK_STRIDE,
        cell_size or HOG_CELL_SIZE,
        nbins or HOG_NBINS,
    )


def load_images_and_extract_hog(base_dir, hog=None, use_roi_subfolder=False):
    """
    Wczytuje obrazy z podkatalogów klasy i ekstrahuje cechy HOG.

    Oczekiwana struktura katalogów:
        base_dir/
            2/          ← ID klasy
                00001.png
                00002.png
            14/
                ...

    Parameters
    ----------
    base_dir : str
        Ścieżka do katalogu z danymi (np. 'processed_data/train').
    hog : cv2.HOGDescriptor, optional
        Deskryptor HOG do użycia. Jeśli None, tworzony jest domyślny.
    use_roi_subfolder : bool, optional
        Czy szukać obrazów w podfolderze 'roi_color' każdej klasy.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list)
        - X_features: macierz cech HOG, kształt (n_samples, n_features)
        - y_labels: wektor etykiet klas, kształt (n_samples,)
        - file_paths: lista ścieżek do plików źródłowych
    """
    if hog is None:
        hog = get_hog_descriptor()

    X_features = []
    y_labels = []
    file_paths = []

    print(f"Wczytywanie danych z: {base_dir}...")

    class_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

    for folder in class_folders:
        # Obsługa nazw folderów typu "12_extracted"
        class_id = int(os.path.basename(folder).split('_')[0])

        search_dir = os.path.join(folder, 'roi_color') if use_roi_subfolder else folder
        image_paths = glob.glob(os.path.join(search_dir, '*.png'))

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            hog_vector = hog.compute(img).flatten()
            X_features.append(hog_vector)
            y_labels.append(class_id)
            file_paths.append(img_path)

    return np.array(X_features), np.array(y_labels), file_paths
