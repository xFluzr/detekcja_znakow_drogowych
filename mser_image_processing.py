import cv2
import numpy as np
import os
import glob


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def process_directory(source_dir, target_dir):
    roi_dir = os.path.join(target_dir, "roi_color")
    edges_dir = os.path.join(target_dir, "roi_edges")
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)

    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.ppm'):
        image_paths.extend(glob.glob(os.path.join(source_dir, ext)))

    if not image_paths:
        print(f"Brak obrazów do przetworzenia w: {source_dir}")
        return

    print(f"Znaleziono {len(image_paths)} obrazów. Uruchamiam detektor MSER...")

    hog = get_hog_descriptor()
    # Inicjalizacja detektora MSER (minimalna plama: 100 px, maksymalna: 50000 px)
    mser = cv2.MSER_create(min_area=100, max_area=50000)

    global_candidate_count = 0
    extracted_features = []
    extracted_labels = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None: continue

        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)

        # MSER działa najlepiej na skali szarości
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. WYKRYWANIE PLAM (BLOBS)
        regions, boundingBoxes = mser.detectRegions(gray)

        local_sign_count = 0

        # Zapamiętujemy już sprawdzone ramki, by uniknąć duplikatów (MSER lubi znaleźć plamę w plamie)
        seen_boxes = []

        for box in boundingBoxes:
            x, y, w, h = box

            # Weryfikacja duplikatów (jeśli już sprawdzaliśmy podobną ramkę, pomijamy)
            is_duplicate = False
            for (sx, sy, sw, sh) in seen_boxes:
                if abs(x - sx) < 5 and abs(y - sy) < 5 and abs(w - sw) < 5 and abs(h - sh) < 5:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            seen_boxes.append((x, y, w, h))

            # 2. Weryfikacja proporcji plamy (znaki to zazwyczaj kwadraty)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 1.5:

                # 3. WERYFIKACJA KOLOREM (Czy w tej stabilnej plamie jest znak?)
                roi_to_check = img[y:y + h, x:x + w]
                if roi_to_check.size == 0: continue

                hsv_roi = cv2.cvtColor(roi_to_check, cv2.COLOR_BGR2HSV)

                # Tolerancyjne maski (sprawdzamy tylko ten wycięty fragment)
                mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 40]), np.array([15, 255, 255]))
                mask_red2 = cv2.inRange(hsv_roi, np.array([165, 50, 40]), np.array([180, 255, 255]))
                mask_blue = cv2.inRange(hsv_roi, np.array([90, 50, 40]), np.array([140, 255, 255]))

                mask_color = mask_red1 | mask_red2 | mask_blue

                # Obliczamy wypełnienie kolorem
                total_pixels = w * h
                color_ratio = cv2.countNonZero(mask_color) / total_pixels

                # Jeśli plama ma sensowny kształt i zawiera co najmniej 15% koloru znaku
                if color_ratio > 0.15:

                    # Wytnij z lekkim marginesem
                    y1 = max(0, y - 3)
                    y2 = min(img.shape[0], y + h + 3)
                    x1 = max(0, x - 3)
                    x2 = min(img.shape[1], x + w + 3)

                    final_roi = img[y1:y2, x1:x2]
                    if final_roi.size == 0: continue

                    roi_resized = cv2.resize(final_roi, (64, 64))
                    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    roi_edges = cv2.Canny(roi_gray, threshold1=100, threshold2=200)
                    hog_vector = hog.compute(roi_resized).flatten()

                    if local_sign_count == 0:
                        save_name = base_name
                    else:
                        save_name = f"{file_name}_{local_sign_count}{file_ext}"

                    cv2.imwrite(os.path.join(roi_dir, save_name), roi_resized)
                    cv2.imwrite(os.path.join(edges_dir, save_name), roi_edges)

                    extracted_features.append(hog_vector)
                    extracted_labels.append(save_name)

                    local_sign_count += 1
                    global_candidate_count += 1

    if extracted_features:
        np.save(os.path.join(target_dir, "hog_features.npy"), np.array(extracted_features))
        np.save(os.path.join(target_dir, "file_names.npy"), np.array(extracted_labels))

    print(f"Zakończono! Zapisano {global_candidate_count} obiektów.")



if __name__ == "__main__":

    classes = [2, 12, 14, 17, 35]

    for class_id in classes:
        SOURCE_DIRECTORY = f'processed_data/test/{class_id}'
        TARGET_DIRECTORY = f'processed_data/processed_test/{class_id}'
        process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY)

    # SOURCE_DIRECTORY = 'testowe_znaki'
    # TARGET_DIRECTORY = 'testowe_znaki/roi'
    # process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY)