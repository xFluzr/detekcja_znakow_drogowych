"""
visualize_hog.py — Wizualizacja deskryptora HOG nałożonego na obraz znaku.

Generuje obrazy pokazujące jak deskryptor HOG "widzi" znak drogowy,
co jest przydatne do prezentacji i raportu technicznego.

Użycie:
    python visualize_hog.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import hog as skimage_hog
from skimage import exposure
from utils import TARGET_CLASSES, CLASS_LABELS

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def visualize_hog_for_class(class_id, data_dir='processed_data/train'):
    """
    Generuje wizualizację HOG dla przykładowego obrazu z danej klasy.

    Parameters
    ----------
    class_id : int
        ID klasy znaku (np. 14 dla STOP).
    data_dir : str
        Ścieżka do katalogu z danymi treningowymi.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Oryginalny obraz i wizualizacja HOG.
    """
    class_dir = os.path.join(data_dir, str(class_id))
    images = [f for f in os.listdir(class_dir) if f.endswith('.png')]

    if not images:
        return None, None

    img_path = os.path.join(class_dir, images[0])
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Obliczenie HOG z wizualizacją (scikit-image)
    features, hog_image = skimage_hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )

    # Poprawa kontrastu wizualizacji
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return img_rgb, hog_image_rescaled


def generate_hog_grid():
    """Generuje siatkę wizualizacji HOG dla wszystkich 5 klas."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for idx, class_id in enumerate(TARGET_CLASSES):
        img_rgb, hog_img = visualize_hog_for_class(class_id)

        if img_rgb is None:
            continue

        label = CLASS_LABELS.get(class_id, str(class_id))
        short_label = label.split('(')[0].strip() if '(' in label else label

        # Pierwszy rząd: oryginały
        axes[0, idx].imshow(img_rgb)
        axes[0, idx].set_title(f"Klasa {class_id}\n{short_label}", fontsize=9)
        axes[0, idx].axis('off')

        # Drugi rząd: wizualizacja HOG
        axes[1, idx].imshow(hog_img, cmap='gray')
        axes[1, idx].set_title("HOG", fontsize=9)
        axes[1, idx].axis('off')

    axes[0, 0].set_ylabel("Oryginał", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Deskryptor HOG", fontsize=11, fontweight='bold')

    fig.suptitle("Wizualizacja deskryptora HOG dla 5 klas znaków", fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'hog_visualization.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Zapisano: {path}")


if __name__ == "__main__":
    generate_hog_grid()
    print("Wizualizacja HOG zakończona!")
