import os
import cv2
import numpy as np
import glob
import shutil
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Etykiety klas (0 = "inne" / odrzucenie)
CLASS_NAMES = {
    0:  "inne (odrzucenie)",
    2:  "50 km/h",
    12: "Pierwszeństwo",
    14: "STOP",
    17: "Zakaz wjazdu",
    35: "Nakaz na wprost",
}


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def load_images_and_extract_hog(base_dir):
    hog = get_hog_descriptor()
    X_features, y_labels, file_paths = [], [], []

    print(f"Wczytywanie danych z: {base_dir}...")
    class_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

    for folder in class_folders:
        class_id = int(os.path.basename(folder).split('_')[0])
        image_paths = glob.glob(os.path.join(folder, '*.png'))

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            X_features.append(hog.compute(img).flatten())
            y_labels.append(class_id)
            file_paths.append(img_path)

    return np.array(X_features), np.array(y_labels), file_paths


if __name__ == "__main__":
    # ==========================================
    # ETAP 1: Przygotowanie Danych
    # ==========================================
    TRAIN_DIR = 'processed_data/train'
    TEST_DIR = 'processed_data/test'

    X_train, y_train, train_paths = load_images_and_extract_hog(TRAIN_DIR)
    print(f"Dane treningowe: {X_train.shape[0]} obrazów.")
    X_test, y_test, test_paths = load_images_and_extract_hog(TEST_DIR)
    print(f"Dane testowe: {X_test.shape[0]} obrazów.\n")

    unique, counts = np.unique(y_train, return_counts=True)
    print("Rozkład klas (train): "
          + ", ".join(f"{CLASS_NAMES.get(c, c)}={n}" for c, n in zip(unique, counts)) + "\n")

    # ==========================================
    # ETAP 2: Trening Modelu SVM (6 klas, z klasą odrzucenia)
    # ==========================================
    # class_weight='balanced' kompensuje liczniejszą klasę "inne",
    # żeby model nie faworyzował odrzucania kosztem trafień.
    print("Rozpoczynam trenowanie klasyfikatora SVM (6 klas + odrzucenie)...")
    svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)

    joblib.dump(svm_model, 'svm_model.joblib')
    print("Trening zakończony! Model zapisano jako 'svm_model.joblib'.\n")

    # ==========================================
    # ETAP 3: Ewaluacja
    # ==========================================
    print("Testowanie na zbiorze testowym...")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ogólna dokładność (Accuracy): {accuracy * 100:.2f}%\n")

    labels_sorted = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[c] for c in labels_sorted]
    print("Szczegółowy raport z klasyfikacji:")
    print(classification_report(y_test, y_pred, labels=labels_sorted,
                                target_names=target_names, zero_division=0))

    # --- Skuteczność odrzucania znaków spoza zestawu ---
    other_mask = (y_test == 0)
    if other_mask.sum() > 0:
        correctly_rejected = np.sum(y_pred[other_mask] == 0)
        print(f"Odrzucanie 'inne': {correctly_rejected}/{other_mask.sum()} "
              f"({correctly_rejected / other_mask.sum() * 100:.1f}%) znaków spoza zestawu "
              f"poprawnie odrzuconych.\n")

    # --- Macierz pomyłek ---
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Macierz Pomyłek Klasyfikatora SVM (5 klas + odrzucenie)")
    plt.ylabel('Rzeczywista Klasa Znaku')
    plt.xlabel('Przewidziana Klasa Znaku')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('macierz_pomylek.png')
    plt.close()
    print("-> Zapisano 'macierz_pomylek.png'\n")

    # ==========================================
    # ETAP 4: Error Tracking (zapis pomyłek)
    # ==========================================
    ERRORS_DIR = 'processed_data/errors'
    if os.path.exists(ERRORS_DIR):
        shutil.rmtree(ERRORS_DIR)
    os.makedirs(ERRORS_DIR, exist_ok=True)

    error_count = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            base_name = os.path.basename(test_paths[i])
            new_filename = f"True_{y_test[i]}_Pred_{y_pred[i]}_{base_name}"
            shutil.copy2(test_paths[i], os.path.join(ERRORS_DIR, new_filename))
            error_count += 1

    print(f"Model pomylił się {error_count} razy.")
    print(f"Błędne klasyfikacje skopiowano do: {ERRORS_DIR}")

    # --- Kolaż błędów (do 8 obrazków, uzupełnia puste miejsca) ---
    error_files = [f for f in os.listdir(ERRORS_DIR) if f.endswith('.png')][:8]
    if error_files:
        tiles = []
        for f in error_files:
            img = cv2.imread(os.path.join(ERRORS_DIR, f))
            tiles.append(cv2.resize(img, (64, 64)))
        while len(tiles) < 8:
            tiles.append(np.zeros((64, 64, 3), dtype=np.uint8))
        rows = [np.hstack(tiles[i * 4:(i + 1) * 4]) for i in range(2)]
        cv2.imwrite('bledy_modelu.png', np.vstack(rows))
        print("-> Zapisano 'bledy_modelu.png' (kolaż 2x4)")
