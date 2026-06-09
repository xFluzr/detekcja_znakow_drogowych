"""
Trenuje YOLOv8n na zbiorze GTSRB (5 klas), trenuje SVM+HOG na tych samych
klasach i drukuje tabelę porównawczą wyników.

Kolejność uruchamiania:
    1. python database-filter.py          (przetwarza dane SVM)
    2. python yolo_prepare_dataset.py     (przygotowuje dataset YOLO)
    3. python yolo_train_and_compare.py   (trenuje oba modele i porównuje)
"""

import os
import time
import glob
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

TARGET_CLASSES = [2, 12, 14, 17, 35]
CLASS_NAMES_MAP = {
    2:  'speed_limit_50',
    12: 'priority_road',
    14: 'stop',
    17: 'no_entry',
    35: 'ahead_only',
}
SORTED_NAMES = [CLASS_NAMES_MAP[c] for c in sorted(TARGET_CLASSES)]


# ─────────────────────────────────────────────
#  HOG + SVM
# ─────────────────────────────────────────────

def get_hog():
    return cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def load_hog_data(base_dir):
    hog = get_hog()
    X, y = [], []
    for folder in os.scandir(base_dir):
        if not folder.is_dir():
            continue
        try:
            class_id = int(os.path.basename(folder.path).split('_')[0])
        except ValueError:
            continue
        # Pomijamy klasę 0 ("inne") – YOLO wykrywa tylko 5 docelowych klas,
        # więc porównanie SVM vs YOLO musi dotyczyć tych samych 5 klas.
        if class_id not in TARGET_CLASSES:
            continue
        for img_path in glob.glob(os.path.join(folder.path, '*.png')):
            img = cv2.imread(img_path)
            if img is None:
                continue
            X.append(hog.compute(img).flatten())
            y.append(class_id)
    return np.array(X), np.array(y)


def run_svm():
    print("\n" + "=" * 55)
    print("ETAP 1 / 2 — HOG + SVM")
    print("=" * 55)

    train_dir = 'processed_data/train'
    test_dir  = 'processed_data/test'

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print("BŁĄD: Brak katalogu processed_data/. Uruchom najpierw database-filter.py")
        return None

    print("Ładowanie cech HOG...")
    X_train, y_train = load_hog_data(train_dir)
    X_test,  y_test  = load_hog_data(test_dir)
    print(f"Train: {len(X_train)} próbek  |  Test: {len(X_test)} próbek")

    print("Trening SVM (kernel=linear)...")
    t0 = time.time()
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(X_test)
    infer_time_total = time.time() - t0
    infer_ms = infer_time_total / len(X_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    print(f"\nAccuracy (macro):  {acc  * 100:.2f}%")
    print(f"Precision (macro): {prec * 100:.2f}%")
    print(f"Recall    (macro): {rec  * 100:.2f}%")
    print(f"F1        (macro): {f1   * 100:.2f}%")
    print(f"Czas treningu:     {train_time:.1f} s")
    print(f"Inferencja/obraz:  {infer_ms:.2f} ms")
    print("\nSzczegółowy raport:")
    print(classification_report(y_test, y_pred, target_names=SORTED_NAMES, zero_division=0))

    return {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'train_time': train_time,
        'infer_ms':   infer_ms,
    }


# ─────────────────────────────────────────────
#  YOLOv8
# ─────────────────────────────────────────────

def run_yolo():
    print("\n" + "=" * 55)
    print("ETAP 2 / 2 — YOLOv8n (detekcja end-to-end)")
    print("=" * 55)

    data_yaml = os.path.join('yolo_dataset', 'data.yaml')
    if not os.path.isfile(data_yaml):
        print("BŁĄD: Brak yolo_dataset/data.yaml. Uruchom najpierw yolo_prepare_dataset.py")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print("BŁĄD: Brak biblioteki ultralytics. Zainstaluj: pip install ultralytics")
        return None

    print("Pobieranie wag YOLOv8n i trenowanie...")
    model = YOLO('yolov8n.pt')

    t0 = time.time()
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=64,
        batch=64,
        project='yolo_runs',
        name='gtsrb_5cls',
        exist_ok=True,
        verbose=False,
        plots=True,
    )
    train_time = time.time() - t0

    print("\nEwaluacja na zbiorze testowym...")
    metrics = model.val(data=data_yaml, verbose=False)

    map50  = metrics.box.map50
    map50_95 = metrics.box.map
    prec   = metrics.box.mp
    rec    = metrics.box.mr
    f1     = 2 * prec * rec / (prec + rec + 1e-9)

    # Czas inferencji – jeden obraz, z wyników walidacji
    speed  = metrics.speed           # słownik: preprocess / inference / postprocess
    infer_ms = speed.get('inference', 0.0)

    print(f"\nmAP@0.5:           {map50    * 100:.2f}%")
    print(f"mAP@0.5:0.95:      {map50_95 * 100:.2f}%")
    print(f"Precision (macro): {prec     * 100:.2f}%")
    print(f"Recall    (macro): {rec      * 100:.2f}%")
    print(f"F1        (macro): {f1       * 100:.2f}%")
    print(f"Czas treningu:     {train_time:.1f} s")
    print(f"Inferencja/obraz:  {infer_ms:.2f} ms")
    print(f"\nWyniki i wykresy: yolo_runs/gtsrb_5cls/")

    return {
        'map50':     map50,
        'map50_95':  map50_95,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'train_time': train_time,
        'infer_ms':   infer_ms,
    }


# ─────────────────────────────────────────────
#  Tabela porównawcza
# ─────────────────────────────────────────────

def print_comparison(svm_res, yolo_res):
    print("\n" + "=" * 65)
    print(" PORÓWNANIE: HOG + SVM  vs  YOLOv8n")
    print("=" * 65)

    W = 32
    print(f"{'Metryka':<{W}} {'HOG+SVM':>14}   {'YOLOv8n':>14}")
    print("-" * 65)

    def fmt(val, pct=True):
        if val is None:
            return "—".rjust(14)
        return (f"{val*100:>13.2f}%" if pct else f"{val:>13.2f} ms")

    svm_acc  = svm_res['accuracy']  if svm_res  else None
    svm_prec = svm_res['precision'] if svm_res  else None
    svm_rec  = svm_res['recall']    if svm_res  else None
    svm_f1   = svm_res['f1']        if svm_res  else None
    svm_tr   = svm_res['train_time'] if svm_res else None
    svm_inf  = svm_res['infer_ms']  if svm_res  else None

    y_map50  = yolo_res['map50']    if yolo_res else None
    y_prec   = yolo_res['precision'] if yolo_res else None
    y_rec    = yolo_res['recall']   if yolo_res else None
    y_f1     = yolo_res['f1']       if yolo_res else None
    y_tr     = yolo_res['train_time'] if yolo_res else None
    y_inf    = yolo_res['infer_ms'] if yolo_res else None

    rows = [
        ("Accuracy / mAP@0.5",     svm_acc,  y_map50,  True),
        ("Precision (macro)",       svm_prec, y_prec,   True),
        ("Recall    (macro)",       svm_rec,  y_rec,    True),
        ("F1        (macro)",       svm_f1,   y_f1,     True),
        ("Czas treningu (s)",       svm_tr,   y_tr,     False),
        ("Inferencja / obraz (ms)", svm_inf,  y_inf,    False),
    ]

    for label, sv, yv, pct in rows:
        sv_str = fmt(sv, pct) if sv is not None else "—".rjust(14)
        yv_str = fmt(yv, pct) if yv is not None else "—".rjust(14)
        print(f"{label:<{W}} {sv_str}   {yv_str}")

    print("=" * 65)
    print()
    print("Uwagi:")
    print("  • HOG+SVM działa na gotowych cropach (64×64) — sama KLASYFIKACJA.")
    print("  • YOLOv8n wykrywa znaki na oryginalnych obrazach — DETEKCJA end-to-end.")
    print("  • mAP@0.5 ≠ Accuracy: mAP wymaga IoU≥0.5 z ground-truth bbox.")
    print()

    # Zapis do pliku
    out_path = 'comparison_results.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("HOG+SVM vs YOLOv8n — wyniki\n\n")
        for label, sv, yv, pct in rows:
            sv_str = f"{sv*100:.2f}%" if (sv is not None and pct) else (f"{sv:.2f} ms" if sv else "—")
            yv_str = f"{yv*100:.2f}%" if (yv is not None and pct) else (f"{yv:.2f} ms" if yv else "—")
            f.write(f"{label:<32} SVM: {sv_str:>10}   YOLO: {yv_str:>10}\n")
    print(f"Wyniki zapisano do: {out_path}")


if __name__ == '__main__':
    svm_results  = run_svm()
    yolo_results = run_yolo()
    print_comparison(svm_results, yolo_results)
