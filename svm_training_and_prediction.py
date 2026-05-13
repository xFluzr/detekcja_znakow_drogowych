import os
import cv2
import numpy as np
import glob
import shutil
import joblib  # Do zapisu wytrenowanego modelu
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def load_images_and_extract_hog(base_dir, use_roi_subfolder=False):
    hog = get_hog_descriptor()
    X_features = []
    y_labels = []
    file_paths = []  # Zbieramy ścieżki do obrazków!

    print(f"Wczytywanie danych z: {base_dir}...")

    class_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

    for folder in class_folders:
        # Pamiętamy o poprawce split('_') dla folderów np. "12_extracted"
        class_id = int(os.path.basename(folder).split('_')[0])

        search_dir = os.path.join(folder, 'roi_color') if use_roi_subfolder else folder
        image_paths = glob.glob(os.path.join(search_dir, '*.png'))

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            hog_vector = hog.compute(img).flatten()

            X_features.append(hog_vector)
            y_labels.append(class_id)
            file_paths.append(img_path)  # Zapisujemy ścieżkę do pliku

    return np.array(X_features), np.array(y_labels), file_paths


if __name__ == "__main__":
    # ==========================================
    # ETAP 1: Przygotowanie Danych
    # ==========================================
    TRAIN_DIR = 'processed_data/train'
    # Wczytujemy 3 zmienne (dodano train_paths)
    X_train, y_train, train_paths = load_images_and_extract_hog(TRAIN_DIR, use_roi_subfolder=False)
    print(f"Dane treningowe załadowane: {X_train.shape[0]} obrazów.\n")

    # TEST_DIR = 'processed_data/processed_test'
    # # Wczytujemy 3 zmienne (dodano test_paths)
    # X_test, y_test, test_paths = load_images_and_extract_hog(TEST_DIR, use_roi_subfolder=True)

    TEST_DIR = 'processed_data/test'
    # Zmieniamy use_roi_subfolder na False (bo nie mamy już podfolderów roi_color)
    X_test, y_test, test_paths = load_images_and_extract_hog(TEST_DIR, use_roi_subfolder=False)
    print(f"Dane testowe załadowane: {X_test.shape[0]} obrazów.\n")

    # ==========================================
    # ETAP 2: Trening Modelu SVM
    # ==========================================
    print("Rozpoczynam trenowanie klasyfikatora SVM...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # --- ZAPIS MODELU DLA APLIKACJI WEBOWEJ ---
    joblib.dump(svm_model, 'svm_model.joblib')
    print("Trening zakończony! Model zapisano jako 'svm_model.joblib'.\n")

    # ==========================================
    # ETAP 3: Ewaluacja i Zapisywanie Błędów
    # ==========================================
    print("Testowanie na wyekstrahowanych danych...")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ogólna dokładność (Accuracy): {accuracy * 100:.2f}%\n")
    print("Szczegółowy raport z klasyfikacji:")
    print(classification_report(y_test, y_pred))

    # --- GENEROWANIE MACIERZY POMYŁEK ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    labels = [2, 12, 14, 17, 35]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Macierz Pomyłek Klasyfikatora SVM")
    plt.ylabel('Rzeczywista Klasa Znaku')
    plt.xlabel('Przewidziana Klasa Znaku')
    plt.tight_layout()
    plt.savefig('macierz_pomylek.png')
    plt.close()
    print("-> Zapisano 'macierz_pomylek.png'\n")

    # ---- ZAPISYWANIE BŁĘDNYCH OBRAZKÓW ----
    ERRORS_DIR = 'processed_data/errors'

    # Czyścimy/tworzymy folder na błędy, żeby nie mieszały się ze starymi uruchomieniami
    if os.path.exists(ERRORS_DIR):
        shutil.rmtree(ERRORS_DIR)
    os.makedirs(ERRORS_DIR, exist_ok=True)

    error_count = 0

    # Przechodzimy przez wszystkie wyniki predykcji
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:  # Jeśli prawdziwa etykieta jest inna niż przewidziana
            true_class = y_test[i]
            pred_class = y_pred[i]
            img_path = test_paths[i]

            # Wyciągamy samą nazwę oryginalnego pliku (np. "00012_0.png")
            base_name = os.path.basename(img_path)

            # Tworzymy nową, informacyjną nazwę pliku
            new_filename = f"True_{true_class}_Pred_{pred_class}_{base_name}"
            dest_path = os.path.join(ERRORS_DIR, new_filename)

            # Kopiujemy obrazek do folderu errors
            shutil.copy2(img_path, dest_path)
            error_count += 1

    print(f"\nUWAGA: Model pomylił się {error_count} razy.")
    print(f"Wszystkie błędnie sklasyfikowane zdjęcia zostały skopiowane do folderu: {ERRORS_DIR}")

    # --- GENEROWANIE KOLAŻU BŁĘDÓW ---
    error_images = []
    # Bierzemy dokładnie 8 obrazków, żeby nie było pustych miejsc
    for f in os.listdir(ERRORS_DIR)[:8]:
        if f.endswith('.png'):
            img_path = os.path.join(ERRORS_DIR, f)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            error_images.append(img)
    
    if len(error_images) == 8:
        rows = []
        # Tworzymy 2 rzędy po 4 zdjęcia (2x4 = 8)
        for i in range(2):
            row = np.hstack(error_images[i*4:(i+1)*4])
            rows.append(row)
        collage = np.vstack(rows)
        cv2.imwrite('bledy_modelu.png', collage)
        print("-> Zapisano 'bledy_modelu.png' (kolaż 2x4)")