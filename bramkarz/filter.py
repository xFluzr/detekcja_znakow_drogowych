import os
import glob


def clean_gtsdb_dataset(dataset_dir):
    gt_file_path = os.path.join(dataset_dir, 'gt.txt')
    new_gt_file_path = os.path.join(dataset_dir, 'gt_filtered.txt')

    # 1. Definiujemy pożądane klasy
    # 14: STOP, 13: Ustąp pierwszeństwa, 2: Ograniczenie Prędkości,
    # 17: Zakaz Wjazdu, 15: Zakaz ruchu w obu kierunkach
    target_classes = {2, 13, 14, 15, 17}

    if not os.path.exists(gt_file_path):
        print(f"BŁĄD: Nie znaleziono pliku {gt_file_path}!")
        return

    # Zmienne do śledzenia statystyk
    images_to_keep = set()
    filtered_lines = []

    print("Analiza pliku gt.txt...")

    # 2. Czytamy oryginalny plik z adnotacjami
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(';')
        if len(parts) == 6:
            filename = parts[0]
            class_id = int(parts[5])

            # Jeśli znak na zdjęciu należy do naszych pożądanych klas
            if class_id in target_classes:
                images_to_keep.add(filename)  # Zaznaczamy zdjęcie do ocalenia
                filtered_lines.append(line + '\n')  # Zapisujemy linijkę do nowego pliku

    print(f"Znaleziono {len(images_to_keep)} unikalnych zdjęć zawierających interesujące nas znaki.")

    # 3. Zapisujemy nowy, wyczyszczony plik tekstowy
    with open(new_gt_file_path, 'w') as f:
        f.writelines(filtered_lines)
    print(f"Zapisano nowy plik z adnotacjami: gt_filtered.txt")

    # 4. Usuwanie niepotrzebnych zdjęć .ppm
    print("Rozpoczynam usuwanie niepotrzebnych obrazów...")

    # Szukamy wszystkich zdjęć .ppm w folderze
    all_ppm_files = glob.glob(os.path.join(dataset_dir, '*.ppm'))
    deleted_count = 0
    kept_count = 0

    for ppm_path in all_ppm_files:
        filename = os.path.basename(ppm_path)

        if filename not in images_to_keep:
            # Jeśli zdjęcia nie ma na liście "do ocalenia" -> usuwamy z dysku
            os.remove(ppm_path)
            deleted_count += 1
        else:
            kept_count += 1

    print("-" * 40)
    print("PODSUMOWANIE CZYSZCZENIA:")
    print(f"Usunięto zdjęć: {deleted_count}")
    print(f"Pozostawiono zdjęć: {kept_count}")
    print("Gotowe do dalszej pracy!")


if __name__ == "__main__":
    # PODAJ ŚCIEŻKĘ DO FOLDERU Z ROZPAKOWANYM GTSDB
    # (Tam gdzie leżą pliki 00000.ppm i plik gt.txt)
    DATASET_DIRECTORY = 'FullIJCNN2013'

    clean_gtsdb_dataset(DATASET_DIRECTORY)