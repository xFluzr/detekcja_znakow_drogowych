import pandas as pd
import cv2
import os
import random

SOURCE_DIR = 'archive'
TARGET_DIR = 'processed_data'
TARGET_SIZE = (64, 64)

random.seed(42)

'''
Nasze 5 docelowych klas:
  2  -> Ograniczenie prędkości (50 km/h)
  12 -> Droga z pierwszeństwem
  14 -> STOP
  17 -> Zakaz wjazdu
  35 -> Nakaz jazdy prosto

Dodatkowo tworzymy klasę odrzucenia:
  0  -> "inne" (próbki z pozostałych 38 klas GTSRB)
       Dzięki niej klasyfikator uczy się, jak wygląda znak SPOZA naszego
       zestawu, i potrafi go odrzucić zamiast wpychać do jednej z 5 klas.
'''
TARGET_CLASSES = [14, 12, 2, 17, 35]
OTHER_CLASS_ID = 0

# Ile próbek "inne" wygenerować (zbiór musi być liczny i różnorodny,
# by skutecznie nauczyć model odrzucania – bierzemy ~2x typowej klasy)
N_OTHER_TRAIN = 3000
N_OTHER_TEST = 900


def _save_image(img, dataset_type, class_id, idx):
    class_dir = os.path.join(TARGET_DIR, dataset_type, str(class_id))
    os.makedirs(class_dir, exist_ok=True)
    save_path = os.path.join(class_dir, f"{idx:05d}.png")
    resized = cv2.resize(img, TARGET_SIZE)
    cv2.imwrite(save_path, resized)


def process_target_classes(df, dataset_type):
    """Zapisuje wszystkie obrazy należące do 5 docelowych klas."""
    print(f"[{dataset_type}] Przetwarzam 5 docelowych klas...")
    df_filtered = df[df['ClassId'].isin(TARGET_CLASSES)]

    counters = {c: 0 for c in TARGET_CLASSES}
    for _, row in df_filtered.iterrows():
        img = cv2.imread(os.path.join(SOURCE_DIR, row['Path']))
        if img is None:
            continue
        cls = int(row['ClassId'])
        _save_image(img, dataset_type, cls, counters[cls])
        counters[cls] += 1

    total = sum(counters.values())
    print(f"   -> zapisano {total} obrazów docelowych: "
          + ", ".join(f"kl.{c}={counters[c]}" for c in TARGET_CLASSES))


def process_other_class(df, dataset_type, n_samples):
    """Buduje klasę 'inne' (0) z losowej, zróżnicowanej próbki pozostałych klas."""
    print(f"[{dataset_type}] Buduję klasę 'inne' (odrzucenie)...")
    df_other = df[~df['ClassId'].isin(TARGET_CLASSES)]

    # Równomierne losowanie po wszystkich pozostałych klasach (różnorodność)
    other_class_ids = sorted(df_other['ClassId'].unique())
    per_class = max(1, n_samples // len(other_class_ids))

    rows = []
    for cid in other_class_ids:
        subset = df_other[df_other['ClassId'] == cid]
        take = min(per_class, len(subset))
        rows.extend(subset.sample(n=take, random_state=42).to_dict('records'))

    random.shuffle(rows)
    rows = rows[:n_samples]

    saved = 0
    for row in rows:
        img = cv2.imread(os.path.join(SOURCE_DIR, row['Path']))
        if img is None:
            continue
        _save_image(img, dataset_type, OTHER_CLASS_ID, saved)
        saved += 1

    print(f"   -> zapisano {saved} obrazów klasy 'inne' "
          f"(z {len(other_class_ids)} pozostałych klas GTSRB)")


def process_dataset(csv_filename, dataset_type, n_other):
    print(f"\n=== Zbiór: {dataset_type} ===")
    df = pd.read_csv(os.path.join(SOURCE_DIR, csv_filename))
    process_target_classes(df, dataset_type)
    process_other_class(df, dataset_type, n_other)


if __name__ == "__main__":
    process_dataset('Train.csv', 'train', N_OTHER_TRAIN)
    process_dataset('Test.csv', 'test', N_OTHER_TEST)
    print("\nGotowe! Dane (5 klas + 'inne') zapisano w 'processed_data/'.")
