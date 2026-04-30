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

    print(f"Znaleziono {len(image_paths)} obrazów. Rozpoczynam potok geometryczny...")

    hog = get_hog_descriptor()
    global_candidate_count = 0
    extracted_features = []
    extracted_labels = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None: continue

        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)

        # ==========================================
        # ETAP 1: KRAWĘDZIE I GEOMETRIA
        # ==========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mocny Blur, żeby zlikwidować szum z liści i chropowatość asfaltu
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Detekcja krawędzi Canny'ego
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Dylatacja (pogrubienie krawędzi), aby połączyć ewentualnie przerwane obwódki znaków
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Szukamy konturów na podstawie krawędzi (używamy RETR_LIST, żeby zajrzeć wszędzie)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        local_sign_count = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h

            # 1. Filtrujemy mikroskopijne kropki i gigantyczne budynki
            if 100 < bbox_area < 50000:
                aspect_ratio = float(w) / h

                # 2. Znaki drogowe zawsze pasują do kwadratowej ramki (0.6 - 1.4)
                if 0.4 < aspect_ratio < 2.0:

                    # 3. Analiza geometryczna: Odrzucamy luźne kreski, szukamy wielokątów
                    peri = cv2.arcLength(cnt, True)
                    # Aproksymacja z marginesem błędu 4%
                    approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
                    vertices = len(approx)

                    # Znak to trójkąt (3), ośmiokąt (8) lub koło (zazwyczaj >6 wierzchołków po aproksymacji)
                    if vertices >= 3:

                        # ==========================================
                        # ETAP 2: OSTATECZNA WERYFIKACJA KOLOREM
                        # ==========================================
                        # Wycinamy ten mały obrys ze zdjęciado sprawdzenia
                        roi_to_check = img[y:y + h, x:x + w]
                        if roi_to_check.size == 0: continue

                        # Konwertujemy tylko ten maleńki kwadracik na HSV
                        hsv_roi = cv2.cvtColor(roi_to_check, cv2.COLOR_BGR2HSV)

                        # Definiujemy bardzo tolerancyjne progi (bo już wiemy, że kształt się zgadza!)
                        mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 40]), np.array([15, 255, 255]))
                        mask_red2 = cv2.inRange(hsv_roi, np.array([165, 50, 40]), np.array([180, 255, 255]))
                        mask_red = mask_red1 | mask_red2
                        mask_blue = cv2.inRange(hsv_roi, np.array([90, 50, 40]), np.array([140, 255, 255]))

                        # Liczymy jaki procent kwadracika stanowi dany kolor
                        total_pixels = w * h
                        red_ratio = cv2.countNonZero(mask_red) / total_pixels
                        blue_ratio = cv2.countNonZero(mask_blue) / total_pixels

                        # Jeśli w tym geometrycznym kształcie jest chociaż 10% czerwonego LUB niebieskiego
                        if red_ratio > 0.10 or blue_ratio > 0.10:

                            # MAMY ZNAK! Wycinamy go bezpiecznie z 2px marginesem
                            y1 = max(0, y - 2)
                            y2 = min(img.shape[0], y + h + 2)
                            x1 = max(0, x - 2)
                            x2 = min(img.shape[1], x + w + 2)

                            final_roi = img[y1:y2, x1:x2]
                            if final_roi.size == 0: continue

                            # Przygotowanie danych dla modelu SVM
                            roi_resized = cv2.resize(final_roi, (64, 64))
                            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                            roi_edges = cv2.Canny(roi_gray, threshold1=100, threshold2=200)
                            hog_vector = hog.compute(roi_resized).flatten()

                            # Zapis
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

    # Zapis cech
    if extracted_features:
        np.save(os.path.join(target_dir, "hog_features.npy"), np.array(extracted_features))
        np.save(os.path.join(target_dir, "file_names.npy"), np.array(extracted_labels))

    print(f"Zakończono! Zapisano {global_candidate_count} poprawnie wyodrębnionych obiektów.")


if __name__ == "__main__":

    classes = [2, 12, 14, 17, 35]

    for class_id in classes:
        SOURCE_DIRECTORY = f'processed_data/test/{class_id}'
        TARGET_DIRECTORY = f'processed_data/processed_test/{class_id}'
        process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY)

    # SOURCE_DIRECTORY = 'testowe_znaki'
    # TARGET_DIRECTORY = 'testowe_znaki/roi'
    # process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY)