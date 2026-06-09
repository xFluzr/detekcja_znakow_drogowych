"""
experiments.py — Automatyczne eksperymenty porównawcze dla projektu detekcji znaków.

Generuje wykresy i tabele porównujące:
1. Kernele SVM (linear, rbf, poly)
2. Wpływ parametru C
3. Wpływ parametrów HOG (cell_size, nbins)
4. Cross-validation (5-fold)

Wyniki zapisywane są do folderu 'results/'.

Użycie:
    python experiments.py
"""

import os
import time
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from utils import get_hog_descriptor, load_images_and_extract_hog, TARGET_CLASSES

# ============================================================
# KONFIGURACJA
# ============================================================
TRAIN_DIR = 'processed_data/train'
TEST_DIR = 'processed_data/test'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Etykiety klas do macierzy pomyłek
CLASS_IDS = TARGET_CLASSES


def save_csv(filename, header, rows):
    """Zapisuje wyniki eksperymentu do pliku CSV."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  -> Zapisano: {path}")


def plot_confusion_matrix(y_true, y_pred, title, filename, normalize=False):
    """
    Generuje i zapisuje macierz pomyłek jako PNG.

    Parameters
    ----------
    y_true : array-like
        Prawdziwe etykiety.
    y_pred : array-like
        Przewidziane etykiety.
    title : str
        Tytuł wykresu.
    filename : str
        Nazwa pliku wynikowego.
    normalize : bool
        Czy normalizować macierz (wartości procentowe).
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_IDS, yticklabels=CLASS_IDS)
    plt.title(title)
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidziana klasa')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Zapisano: {path}")


# ============================================================
# EKSPERYMENT 1: Porównanie kerneli SVM
# ============================================================
def experiment_kernels(X_train, y_train, X_test, y_test):
    """Porównuje kernele SVM: linear, rbf, poly."""
    print("\n" + "=" * 60)
    print("EKSPERYMENT 1: Porównanie kerneli SVM")
    print("=" * 60)

    kernels = ['linear', 'rbf', 'poly']
    rows = []

    for kernel in kernels:
        print(f"\n  Trenuję SVM z kernel='{kernel}'...")
        t0 = time.time()
        model = SVC(kernel=kernel, C=1.0, random_state=42)
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        print(f"  Kernel={kernel}: accuracy={acc*100:.2f}%, "
              f"czas treningu={train_time:.1f}s, czas predykcji={pred_time:.3f}s")

        rows.append([kernel, f"{acc*100:.2f}", f"{train_time:.2f}", f"{pred_time:.3f}"])

        # Macierz pomyłek dla każdego kernela
        plot_confusion_matrix(y_test, y_pred,
                              f"Macierz pomyłek — SVM (kernel={kernel})",
                              f"cm_kernel_{kernel}.png")

    save_csv("exp1_kernels.csv",
             ["Kernel", "Accuracy (%)", "Czas treningu (s)", "Czas predykcji (s)"],
             rows)

    # Wykres słupkowy
    labels = [r[0] for r in rows]
    accs = [float(r[1]) for r in rows]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accs, color=['#3b82f6', '#ef4444', '#22c55e'], edgecolor='black')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
    plt.title("Porównanie kerneli SVM — Accuracy", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_kernels_bar.png"), dpi=150)
    plt.close()
    print(f"  -> Zapisano: {RESULTS_DIR}/exp1_kernels_bar.png")


# ============================================================
# EKSPERYMENT 2: Wpływ parametru C
# ============================================================
def experiment_param_c(X_train, y_train, X_test, y_test):
    """Bada wpływ parametru regularyzacji C na accuracy SVM."""
    print("\n" + "=" * 60)
    print("EKSPERYMENT 2: Wpływ parametru C")
    print("=" * 60)

    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    rows = []

    for c in c_values:
        print(f"  C={c}...")
        model = SVC(kernel='linear', C=c, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rows.append([c, f"{acc*100:.2f}"])
        print(f"    Accuracy: {acc*100:.2f}%")

    save_csv("exp2_param_c.csv", ["C", "Accuracy (%)"], rows)

    # Wykres liniowy
    cs = [float(r[0]) for r in rows]
    accs = [float(r[1]) for r in rows]
    plt.figure(figsize=(8, 5))
    plt.semilogx(cs, accs, 'o-', color='#3b82f6', linewidth=2, markersize=8)
    for x, y in zip(cs, accs):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)
    plt.title("Wpływ parametru C na Accuracy (SVM linear)", fontsize=14)
    plt.xlabel("C (skala logarytmiczna)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_param_c.png"), dpi=150)
    plt.close()
    print(f"  -> Zapisano: {RESULTS_DIR}/exp2_param_c.png")


# ============================================================
# EKSPERYMENT 3: Wpływ parametrów HOG
# ============================================================
def experiment_hog_params(y_train_ref, y_test_ref):
    """
    Porównuje różne konfiguracje deskryptora HOG.

    Uwaga: wymaga ponownego wczytania obrazów dla każdej konfiguracji,
    dlatego przyjmuje etykiety referencyjne do weryfikacji.
    """
    print("\n" + "=" * 60)
    print("EKSPERYMENT 3: Wpływ parametrów HOG")
    print("=" * 60)

    configs = [
        {"cell_size": (4, 4),  "nbins": 9,  "label": "cell=4x4, bins=9"},
        {"cell_size": (8, 8),  "nbins": 6,  "label": "cell=8x8, bins=6"},
        {"cell_size": (8, 8),  "nbins": 9,  "label": "cell=8x8, bins=9 (domyślny)"},
        {"cell_size": (8, 8),  "nbins": 12, "label": "cell=8x8, bins=12"},
        {"cell_size": (16, 16), "nbins": 9, "label": "cell=16x16, bins=9"},
    ]

    rows = []

    for cfg in configs:
        cell = cfg["cell_size"]
        nbins = cfg["nbins"]
        label = cfg["label"]
        print(f"\n  Konfig: {label}")

        # Dopasowanie block_size i block_stride do cell_size
        block_size = (cell[0] * 2, cell[1] * 2)
        block_stride = cell

        # Sprawdź czy block_size mieści się w oknie 64x64
        if block_size[0] > 64 or block_size[1] > 64:
            print(f"    POMINIĘTO — block_size {block_size} > okno 64x64")
            continue

        try:
            hog = get_hog_descriptor(
                cell_size=cell, nbins=nbins,
                block_size=block_size, block_stride=block_stride
            )
        except Exception as e:
            print(f"    BŁĄD tworzenia HOG: {e}")
            continue

        X_train, y_train, _ = load_images_and_extract_hog(TRAIN_DIR, hog=hog)
        X_test, y_test, _ = load_images_and_extract_hog(TEST_DIR, hog=hog)

        n_features = X_train.shape[1]
        print(f"    Wymiar cechy: {n_features}")

        model = SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"    Accuracy: {acc*100:.2f}%")

        rows.append([label, n_features, f"{acc*100:.2f}"])

    save_csv("exp3_hog_params.csv",
             ["Konfiguracja HOG", "Wymiar cechy", "Accuracy (%)"], rows)

    # Wykres
    if rows:
        labels = [r[0] for r in rows]
        accs = [float(r[2]) for r in rows]
        dims = [int(r[1]) for r in rows]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = range(len(labels))
        bars = ax1.bar(x, accs, color='#3b82f6', alpha=0.8, label='Accuracy')
        ax1.set_ylabel('Accuracy (%)', color='#3b82f6')
        ax1.set_ylim(0, 105)

        ax2 = ax1.twinx()
        ax2.plot(x, dims, 'o-', color='#ef4444', linewidth=2, markersize=8, label='Wymiar cechy')
        ax2.set_ylabel('Wymiar wektora cech', color='#ef4444')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax1.set_title("Wpływ parametrów HOG na Accuracy i wymiar cechy", fontsize=13)

        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')

        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "exp3_hog_params.png"), dpi=150)
        plt.close()
        print(f"  -> Zapisano: {RESULTS_DIR}/exp3_hog_params.png")


# ============================================================
# EKSPERYMENT 4: Cross-validation (5-fold)
# ============================================================
def experiment_cross_validation(X_train, y_train):
    """Wykonuje 5-fold cross-validation na danych treningowych."""
    print("\n" + "=" * 60)
    print("EKSPERYMENT 4: Cross-validation (5-fold)")
    print("=" * 60)

    kernels = ['linear', 'rbf']
    rows = []

    for kernel in kernels:
        print(f"  Kernel='{kernel}' — 5-fold CV...")
        model = SVC(kernel=kernel, C=1.0, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean = scores.mean() * 100
        std = scores.std() * 100
        print(f"    Wynik: {mean:.2f}% ± {std:.2f}%")
        print(f"    Poszczególne foldy: {[f'{s*100:.1f}%' for s in scores]}")
        rows.append([kernel, f"{mean:.2f}", f"{std:.2f}",
                      ", ".join([f"{s*100:.1f}" for s in scores])])

    save_csv("exp4_cross_validation.csv",
             ["Kernel", "Średnia Accuracy (%)", "Odch. std. (%)", "Wyniki foldów (%)"],
             rows)


# ============================================================
# EKSPERYMENT 5: Normalized Confusion Matrix (najlepszy model)
# ============================================================
def experiment_best_model_analysis(X_train, y_train, X_test, y_test):
    """Szczegółowa analiza najlepszego modelu (linear, C=1.0)."""
    print("\n" + "=" * 60)
    print("EKSPERYMENT 5: Szczegółowa analiza najlepszego modelu")
    print("=" * 60)

    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Raport klasyfikacji
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in CLASS_IDS],
                                   output_dict=True)
    print("\n  Raport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in CLASS_IDS]))

    # Zapis per-class metryk
    rows = []
    for cls_id in CLASS_IDS:
        key = str(cls_id)
        if key in report:
            r = report[key]
            rows.append([cls_id, f"{r['precision']*100:.1f}", f"{r['recall']*100:.1f}",
                         f"{r['f1-score']*100:.1f}", int(r['support'])])
    save_csv("exp5_per_class_metrics.csv",
             ["Klasa", "Precision (%)", "Recall (%)", "F1-score (%)", "Liczba próbek"],
             rows)

    # Macierz pomyłek — zwykła i znormalizowana
    plot_confusion_matrix(y_test, y_pred,
                          "Macierz pomyłek (wartości bezwzględne)",
                          "exp5_cm_absolute.png", normalize=False)
    plot_confusion_matrix(y_test, y_pred,
                          "Macierz pomyłek (znormalizowana)",
                          "exp5_cm_normalized.png", normalize=True)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  EKSPERYMENTY — Detekcja Znaków Drogowych")
    print(f"  Wyniki będą zapisywane w: {RESULTS_DIR}/")
    print("=" * 60)

    # Wczytanie danych z domyślnym HOG
    X_train, y_train, _ = load_images_and_extract_hog(TRAIN_DIR)
    X_test, y_test, _ = load_images_and_extract_hog(TEST_DIR)
    print(f"\nDane: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"wymiar cechy: {X_train.shape[1]}\n")

    # Uruchom eksperymenty
    experiment_kernels(X_train, y_train, X_test, y_test)
    experiment_param_c(X_train, y_train, X_test, y_test)
    experiment_hog_params(y_train, y_test)
    experiment_cross_validation(X_train, y_train)
    experiment_best_model_analysis(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("  WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE!")
    print(f"  Sprawdź folder: {RESULTS_DIR}/")
    print("=" * 60)
