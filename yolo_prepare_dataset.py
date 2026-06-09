import os
import shutil
import pandas as pd
import cv2

SOURCE_DIR = 'archive'
YOLO_DIR = 'yolo_dataset'

TARGET_CLASSES = [2, 12, 14, 17, 35]
CLASS_NAMES = {
    2:  'speed_limit_50',
    12: 'priority_road',
    14: 'stop',
    17: 'no_entry',
    35: 'ahead_only',
}
CLASS_MAP = {cls: idx for idx, cls in enumerate(sorted(TARGET_CLASSES))}


def prepare_split(csv_filename, split_name):
    images_dir = os.path.join(YOLO_DIR, 'images', split_name)
    labels_dir = os.path.join(YOLO_DIR, 'labels', split_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    csv_path = os.path.join(SOURCE_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    df = df[df['ClassId'].isin(TARGET_CLASSES)]

    skipped = 0
    count = 0

    for _, row in df.iterrows():
        img_path = os.path.join(SOURCE_DIR, row['Path'])
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        h_img, w_img = img.shape[:2]
        x1 = int(row['Roi.X1'])
        y1 = int(row['Roi.Y1'])
        x2 = int(row['Roi.X2'])
        y2 = int(row['Roi.Y2'])

        # Normalizacja do formatu YOLO: cx cy w h (0–1)
        cx = (x1 + x2) / 2 / w_img
        cy = (y1 + y2) / 2 / h_img
        bw = (x2 - x1) / w_img
        bh = (y2 - y1) / h_img

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        class_idx = CLASS_MAP[int(row['ClassId'])]

        # Unikalna nazwa pliku (zachowujemy strukturę ścieżki w nazwie)
        base_name = row['Path'].replace('/', '_').replace('\\', '_')
        stem = os.path.splitext(base_name)[0]

        dst_img = os.path.join(images_dir, base_name)
        dst_lbl = os.path.join(labels_dir, stem + '.txt')

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, 'w') as f:
            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        count += 1

    print(f"[{split_name:5s}] skopiowano {count} obrazów  (pominięto: {skipped})")
    return count


def write_yaml():
    abs_yolo = os.path.abspath(YOLO_DIR).replace('\\', '/')
    names_list = [CLASS_NAMES[c] for c in sorted(TARGET_CLASSES)]

    yaml_lines = [
        f"path: {abs_yolo}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(TARGET_CLASSES)}",
        f"names: {names_list}",
    ]

    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines) + '\n')

    print(f"\nZapisano: {yaml_path}")
    print(f"Klasy: {names_list}")


if __name__ == '__main__':
    print("=== Przygotowanie datasetu YOLO z GTSRB ===\n")
    n_train = prepare_split('Train.csv', 'train')
    n_val   = prepare_split('Test.csv',  'val')
    write_yaml()
    print(f"\nŁącznie: {n_train + n_val} obrazów")
    print("\nGotowe! Uruchom teraz: python yolo_train_and_compare.py")
