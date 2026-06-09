"""
Generuje brakujące obrazy potrzebne do raportu LaTeX.
Wymaga wcześniej uruchomionego database-filter.py (folder processed_data/).
"""
import cv2
import numpy as np
import glob
import os
import random

random.seed(42)
np.random.seed(42)

DATA_DIR = 'processed_data/test'
TARGET_CLASSES = [2, 12, 14, 17, 35]
CLASS_LABELS = {
    2:  '2: 50 km/h',
    12: '12: Pierwszeństwo',
    14: '14: STOP',
    17: '17: Zakaz wjazdu',
    35: '35: Nakaz na wprost',
}


def load_sample(class_id, n=4):
    folder = os.path.join(DATA_DIR, str(class_id))
    paths = glob.glob(os.path.join(folder, '*.png'))
    random.shuffle(paths)
    imgs = []
    for p in paths[:n]:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(cv2.resize(img, (64, 64)))
    return imgs


# ─────────────────────────────────────────────────────────
# 1. klasy_gtsrb.png  –  kolaż 5×4 próbek klas
# ─────────────────────────────────────────────────────────
def make_klasy_gtsrb():
    rows = []
    for cls in TARGET_CLASSES:
        samples = load_sample(cls, n=4)
        if len(samples) < 4:
            samples += [np.zeros((64, 64, 3), dtype=np.uint8)] * (4 - len(samples))
        row_img = np.hstack(samples)

        label = CLASS_LABELS[cls]
        label_bar = np.zeros((24, row_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (220, 220, 220), 1, cv2.LINE_AA)
        rows.append(np.vstack([label_bar, row_img]))

    collage = np.vstack(rows)
    cv2.imwrite('klasy_gtsrb.png', collage)
    print("-> Zapisano klasy_gtsrb.png")


# ─────────────────────────────────────────────────────────
# 2. 01_blur.png  –  oryginał vs obraz po rozmyciu Gaussa
# ─────────────────────────────────────────────────────────
def make_blur():
    paths = glob.glob(os.path.join(DATA_DIR, '14', '*.png'))
    if not paths:
        print("POMINIĘTO 01_blur.png – brak obrazów klasy 14")
        return
    img = cv2.imread(paths[min(10, len(paths) - 1)])
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    separator = np.ones((128, 4, 3), dtype=np.uint8) * 200
    combined = np.hstack([gray_bgr, separator, blur_bgr])

    add_caption(combined, "Przed wygładzeniem (skala szarości)", 0, 128)
    add_caption(combined, "Po rozmyciu Gaussa (3x3)", 132, 132 + 128)

    cv2.imwrite('01_blur.png', combined)
    print("-> Zapisano 01_blur.png")


# ─────────────────────────────────────────────────────────
# 3. 02_canny_dylatacja.png  –  krawędzie Canny + dylatacja
# ─────────────────────────────────────────────────────────
def make_canny():
    paths = glob.glob(os.path.join(DATA_DIR, '2', '*.png'))
    if not paths:
        print("POMINIĘTO 02_canny_dylatacja.png – brak obrazów klasy 2")
        return
    img = cv2.imread(paths[min(5, len(paths) - 1)])
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    img_resized = cv2.resize(img, (128, 128))
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    separator = np.ones((128, 4, 3), dtype=np.uint8) * 200
    combined = np.hstack([img_resized, separator, edges_bgr, separator, dilated_bgr])

    add_caption(combined, "Obraz wejściowy", 0, 128)
    add_caption(combined, "Krawędzie Canny", 132, 260)
    add_caption(combined, "Po dylatacji", 264, 392)

    cv2.imwrite('02_canny_dylatacja.png', combined)
    print("-> Zapisano 02_canny_dylatacja.png")


# ─────────────────────────────────────────────────────────
# 4. 03_maska_hsv.png  –  wycięty kandydat + maska HSV
# ─────────────────────────────────────────────────────────
def make_hsv_mask():
    paths = glob.glob(os.path.join(DATA_DIR, '17', '*.png'))
    if not paths:
        print("POMINIĘTO 03_maska_hsv.png – brak obrazów klasy 17")
        return
    img = cv2.imread(paths[min(3, len(paths) - 1)])
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]),   np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 50, 40]), np.array([180, 255, 255]))
    mask_red  = mask_red1 | mask_red2

    mask_red_bgr = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
    masked_img   = cv2.bitwise_and(img, img, mask=mask_red)

    separator = np.ones((128, 4, 3), dtype=np.uint8) * 200
    combined = np.hstack([img, separator, mask_red_bgr, separator, masked_img])

    add_caption(combined, "Oryginalny znak (RGB)", 0, 128)
    add_caption(combined, "Maska czerwieni (HSV)", 132, 260)
    add_caption(combined, "Wynik maskowania", 264, 392)

    cv2.imwrite('03_maska_hsv.png', combined)
    print("-> Zapisano 03_maska_hsv.png")


# ─────────────────────────────────────────────────────────
# 5. 04_mser.png  –  wizualizacja plam MSER
# ─────────────────────────────────────────────────────────
def make_mser():
    paths = glob.glob(os.path.join(DATA_DIR, '35', '*.png'))
    if not paths:
        print("POMINIĘTO 04_mser.png – brak obrazów klasy 35")
        return
    img = cv2.imread(paths[min(20, len(paths) - 1)])
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create(min_area=30, max_area=5000)
    regions, bboxes = mser.detectRegions(gray)

    vis = img.copy()
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(vis, [hull], True, (0, 255, 0), 1)

    vis2 = img.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(vis2, (x, y), (x + w, y + h), (0, 200, 255), 1)

    separator = np.ones((128, 4, 3), dtype=np.uint8) * 200
    combined = np.hstack([img, separator, vis, separator, vis2])

    add_caption(combined, "Obraz wejściowy", 0, 128)
    add_caption(combined, "Regiony MSER (kontury)", 132, 260)
    add_caption(combined, "Ramki bounding box", 264, 392)

    cv2.imwrite('04_mser.png', combined)
    print("-> Zapisano 04_mser.png")


# ─────────────────────────────────────────────────────────
# Pomocnik: pasek tytułowy pod obrazem
# ─────────────────────────────────────────────────────────
def add_caption(img, text, x_start, x_end):
    h = img.shape[0]
    center_x = (x_start + x_end) // 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    tx = center_x - text_size[0] // 2
    cv2.putText(img, text, (tx, h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (80, 80, 200), 1, cv2.LINE_AA)


if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR):
        print(f"BŁĄD: Brak katalogu '{DATA_DIR}'.")
        print("Uruchom najpierw: python database-filter.py")
    else:
        make_klasy_gtsrb()
        make_blur()
        make_canny()
        make_hsv_mask()
        make_mser()
        print("\nGotowe! Wszystkie obrazy zostały wygenerowane.")
