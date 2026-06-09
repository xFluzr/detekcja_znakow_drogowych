"""
Trenuje binarny klasyfikator (bramkarz) do wykrywania "czy to znak?" w sliding window.

Bramkarz to lekki filtr HOG+SVM (probability=True) który w aplikacji GUI
odrzuca okna nie zawierające żadnego znaku, zanim zapyta ekspert-SVM o klasę.

Próbki:
  • Pozytywne (klasa 1) — 64×64 wycinki znaków z processed_data/train/
  • Negatywne (klasa 0) — losowe patche 64×64 z obrazów GTSRB poza obszarem ROI.
    Jeśli brak archiwum, generuje syntetyczny szum jako tło.

Wynik: bramkarz_model.pkl

Kolejność uruchamiania:
  1. python database-filter.py              (tworzy processed_data/)
  2. python train_gatekeeper.py             (tworzy bramkarz_model.pkl)
  3. python svm_training_and_prediction.py  (tworzy svm_model.joblib)
  4. python gui.py
"""

import os
import glob
import random
import numpy as np
import cv2
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report

random.seed(42)
np.random.seed(42)

PROCESSED_TRAIN_DIR = 'processed_data/train'
ARCHIVE_DIR = 'archive'
PATCH_SIZE = (64, 64)
NEG_PER_IMAGE = 5
MAX_NEGATIVES = 5000
TARGET_CLASSES = [2, 12, 14, 17, 35]


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def collect_positive_samples(hog):
    """Wczytuje i oblicza HOG dla wszystkich wycinków znaków (etykieta 1)."""
    features = []
    for class_id in TARGET_CLASSES:
        folder = os.path.join(PROCESSED_TRAIN_DIR, str(class_id))
        for img_path in glob.glob(os.path.join(folder, '*.png')):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, PATCH_SIZE)
            features.append(hog.compute(img).flatten())
    print(f"  Pozytywne próbki (znaki):  {len(features)}")
    return features


def collect_negative_samples(hog, n_max=MAX_NEGATIVES):
    """
    Pobiera losowe patche tła z obrazów GTSRB (etykieta 0).
    Wybiera tylko obrazy spoza naszych 5 klas, by tło było czyste.
    W razie braku archiwum generuje syntetyczny szum pikselowy.
    """
    features = []

    csv_path = os.path.join(ARCHIVE_DIR, 'Train.csv')
    if not os.path.isfile(csv_path):
        print("  OSTRZEŻENIE: Brak archive/Train.csv. Używam syntetycznych patchy negatywnych.")
        for _ in range(n_max):
            noise = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            features.append(hog.compute(noise).flatten())
        print(f"  Negatywne próbki (tło):    {len(features)}")
        return features

    df = pd.read_csv(csv_path)
    df_background = df[~df['ClassId'].isin(TARGET_CLASSES)].sample(
        min(n_max // NEG_PER_IMAGE, len(df)), random_state=42
    )

    pw, ph = PATCH_SIZE
    for _, row in df_background.iterrows():
        if len(features) >= n_max:
            break
        img_path = os.path.join(ARCHIVE_DIR, row['Path'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        roi_x1, roi_y1 = int(row['Roi.X1']), int(row['Roi.Y1'])
        roi_x2, roi_y2 = int(row['Roi.X2']), int(row['Roi.Y2'])

        for _ in range(NEG_PER_IMAGE):
            if w <= pw or h <= ph:
                continue
            x = random.randint(0, w - pw)
            y = random.randint(0, h - ph)

            # Odrzuć patch jeśli mocno zachodzi na ROI znaku (>30% pokrycia)
            overlap_x = max(0, min(x + pw, roi_x2) - max(x, roi_x1))
            overlap_y = max(0, min(y + ph, roi_y2) - max(y, roi_y1))
            if overlap_x * overlap_y > 0.3 * pw * ph:
                continue

            patch = img[y:y + ph, x:x + pw]
            if patch.shape[:2] != (ph, pw):
                continue
            features.append(hog.compute(patch).flatten())

    # Uzupełniamy syntetycznym szumem jeśli zebraliśmy za mało próbek
    while len(features) < min(1000, n_max):
        noise = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        features.append(hog.compute(noise).flatten())

    print(f"  Negatywne próbki (tło):    {len(features)}")
    return features


if __name__ == '__main__':
    print("=== Trenowanie bramkarza (gatekeeper) ===\n")

    if not os.path.isdir(PROCESSED_TRAIN_DIR):
        print(f"BŁĄD: Brak katalogu '{PROCESSED_TRAIN_DIR}'.")
        print("Uruchom najpierw: python database-filter.py")
        raise SystemExit(1)

    hog = get_hog_descriptor()

    print("Zbieranie próbek...")
    pos = collect_positive_samples(hog)
    neg = collect_negative_samples(hog)

    if not pos or not neg:
        print("BŁĄD: Za mało próbek do treningu.")
        raise SystemExit(1)

    X = np.array(pos + neg)
    y = np.array([1] * len(pos) + [0] * len(neg))

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\nPróbki treningowe: {len(X_train)}  |  Walidacyjne: {len(X_val)}")
    print("Trening SVC z jądrem RBF i probability=True ...")

    clf = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\nWyniki na zbiorze walidacyjnym:")
    print(classification_report(y_val, y_pred, target_names=['tło (0)', 'znak (1)']))

    joblib.dump(clf, 'bramkarz_model.pkl')
    print("-> Zapisano bramkarz_model.pkl")
    print("\nGotowe! Możesz teraz uruchomić: python svm_training_and_prediction.py")
