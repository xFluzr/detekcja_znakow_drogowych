import pandas as pd
import cv2
import os


SOURCE_DIR = 'archive'
TARGET_DIR = 'processed_data'
TARGET_SIZE = (64, 64)

'''
14 -> STOP
13 -> Ustąp pierwszeństwa
2  -> Ograniczenie Prędkości
17 -> Zakaz Wjazdu
15 -> Zakaz ruchu w obu kierunkach

'''
# Nasze wybrane 5 klas znaków
TARGET_CLASSES = [14, 12, 2, 17, 35]

def process_dataset(csv_filename, dataset_type, extract_roi=True, resize=True):
    print(f"Rozpoczynam przetwarzanie zbioru: {dataset_type}...")

    csv_path = os.path.join(SOURCE_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    df_filtered = df[df['ClassId'].isin(TARGET_CLASSES)]

    # 3. Wstępna obróbka - iteracja po zdjęciach
    processed_count = 0
    for index, row in df_filtered.iterrows():
        img_path = os.path.join(SOURCE_DIR, row['Path'])

        # Wczytanie obrazu
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Wykadrowanie znaku na podstawie koordynatów ROI (Region of Interest)
        if extract_roi:
            x1, y1 = int(row['Roi.X1']), int(row['Roi.Y1'])
            x2, y2 = int(row['Roi.X2']), int(row['Roi.Y2'])
            cropped_img = img[y1:y2, x1:x2]
        else:
            cropped_img = img

        if resize:
            # 4. Normalizacja - zmiana rozmiaru do 64x64
            resized_img = cv2.resize(cropped_img, TARGET_SIZE)
        else:
            resized_img = img

        # Zapisanie przetworzonego pliku
        class_dir = os.path.join(TARGET_DIR, dataset_type, str(row['ClassId']))
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{processed_count:05d}.png")
        cv2.imwrite(save_path, resized_img)
        processed_count += 1

    print(f"Zakończono! Zapisano {processed_count} obrazów w {TARGET_DIR}/{dataset_type}/\n")


# Uruchomienie funkcji dla zbioru treningowego i testowego
if __name__ == "__main__":
    process_dataset('Train.csv', 'train', extract_roi=False, resize=True)
    process_dataset('Test.csv', 'test', extract_roi=False, resize=True)