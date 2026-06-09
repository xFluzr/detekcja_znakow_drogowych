import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import joblib
import os

# ════════════════════════════════════════════════════════════════
#  KLASY
# ════════════════════════════════════════════════════════════════
# 0 = "inne" (znak spoza naszego zestawu) → odrzucany, nie rysujemy
CLASS_LABELS = {
    2:  "Ograniczenie prędkości (50 km/h)",
    12: "Droga z pierwszeństwem",
    14: "Znak STOP",
    17: "Zakaz wjazdu",
    35: "Nakaz jazdy prosto",
}
REJECT_CLASS = 0

# ════════════════════════════════════════════════════════════════
#  PARAMETRY STROJENIA (zmień i zrestartuj GUI)
# ════════════════════════════════════════════════════════════════
# Minimalna pewność (margines SVM) dla przewidzianej klasy.
# Klasa "inne" robi większość pracy z odrzucaniem; ten próg odcina resztki.
CONFIDENCE_THRESHOLD = 0.3

# Obraz wejściowy o obu wymiarach <= tego progu uznajemy za gotowy wycinek znaku.
DIRECT_CLASSIFY_MAX_DIM = 150

# Detekcja na pełnym zdjęciu (segmentacja kolor + kształt):
MIN_AREA = 600          # minimalne pole kandydata [px]
MAX_AREA_RATIO = 0.6    # maks. pole jako ułamek całego obrazu
MIN_COLOR_RATIO = 0.10  # min. udział barwy znaku w kandydacie

# Progi HSV barw znaków drogowych
HSV_RANGES = [
    (np.array([0,   60, 40]),  np.array([12,  255, 255])),   # czerwony dolny
    (np.array([168, 60, 40]),  np.array([180, 255, 255])),   # czerwony górny
    (np.array([95,  70, 40]),  np.array([135, 255, 255])),   # niebieski
    (np.array([18,  90, 90]),  np.array([32,  255, 255])),   # żółty (pierwszeństwo)
]


def get_hog_descriptor():
    return cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def build_color_mask(img_bgr):
    """Maska binarna pikseli o barwach znaków (czerwony/niebieski/żółty)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def color_ratio(img_bgr):
    """Udział pikseli barwy znaku w wycinku."""
    if img_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    return cv2.countNonZero(mask) / mask.size


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detekcja i Klasyfikacja Znaków Drogowych — HOG + SVM")
        self.root.geometry("900x760")
        self.root.configure(bg="#1e1e2e")

        self.hog = get_hog_descriptor()
        self.model = None

        self.load_model()
        self.setup_ui()

    # ──────────────────────────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        if os.path.exists('svm_model.joblib'):
            try:
                self.model = joblib.load('svm_model.joblib')
                print("Załadowano svm_model.joblib")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować modelu: {e}")
        else:
            messagebox.showwarning(
                "Brak modelu",
                "Brak 'svm_model.joblib'.\nUruchom najpierw:\n"
                "  python database-filter.py\n  python svm_training_and_prediction.py")

    def _predict(self, roi_bgr):
        """Klasyfikuje wycinek 64×64. Zwraca (class_id, confidence)."""
        resized = cv2.resize(roi_bgr, (64, 64))
        hog_vec = self.hog.compute(resized).flatten()
        pred = int(self.model.predict([hog_vec])[0])
        try:
            dec = self.model.decision_function([hog_vec])[0]
            idx = list(self.model.classes_).index(pred)
            conf = float(dec[idx])
        except Exception:
            conf = 1.0
        return pred, conf

    # ──────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────

    def setup_ui(self):
        tk.Label(self.root, text="Detekcja i Klasyfikacja Znaków Drogowych",
                 font=("Helvetica", 16, "bold"), bg="#1e1e2e", fg="white").pack(pady=10)

        tk.Button(self.root, text="Wybierz zdjęcie i Skanuj", font=("Helvetica", 12),
                  bg="#3b82f6", fg="white", activebackground="#2563eb",
                  command=self.open_file).pack(pady=5)

        self.result_label = tk.Label(self.root, text="Oczekuję na zdjęcie…",
                                     font=("Helvetica", 13), bg="#1e1e2e", fg="#94a3b8",
                                     wraplength=860)
        self.result_label.pack(pady=8)

        self.mode_label = tk.Label(self.root, text="", font=("Helvetica", 10, "italic"),
                                   bg="#1e1e2e", fg="#64748b")
        self.mode_label.pack()

        self.image_panel = tk.Label(self.root, bg="#1e1e2e")
        self.image_panel.pack(pady=8)

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.ppm *.bmp")])
        if path:
            self.predict_sign(path)

    # ──────────────────────────────────────────────────────────────
    # Główna logika
    # ──────────────────────────────────────────────────────────────

    def predict_sign(self, file_path):
        if self.model is None:
            self.result_label.config(text="Błąd: brak modelu!", fg="#ef4444")
            return

        self.result_label.config(text="Analizuję obraz…", fg="#f1c40f")
        self.root.update()

        raw = open(file_path, "rb").read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self.result_label.config(text="Nie udało się wczytać obrazu.", fg="#ef4444")
            return

        h0, w0 = img.shape[:2]

        if h0 <= DIRECT_CLASSIFY_MAX_DIM and w0 <= DIRECT_CLASSIFY_MAX_DIM:
            self._direct_mode(img)
        else:
            self._detection_mode(img)

    # ── TRYB 1: gotowy wycinek znaku ─────────────────────────────
    def _direct_mode(self, img):
        self.mode_label.config(text="Tryb: klasyfikacja bezpośrednia (wycinek znaku)")
        pred, conf = self._predict(img)

        display = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

        rejected = (pred == REJECT_CLASS) or (conf < CONFIDENCE_THRESHOLD)
        if rejected:
            self.result_label.config(
                text="Nierozpoznany — znak spoza projektu (nie należy do 5 obsługiwanych klas)",
                fg="#f39c12")
            cv2.rectangle(display, (2, 2), (253, 253), (0, 140, 255), 3)
            display = self._draw_text(display, "NIEROZPOZNANY", (4, 4), (0, 140, 255), 14)
        else:
            label = CLASS_LABELS[pred]
            self.result_label.config(text=f"Rozpoznany znak: {label}  (pewność {conf:.2f})",
                                     fg="#2ecc71")
            cv2.rectangle(display, (2, 2), (253, 253), (0, 255, 0), 3)
            display = self._draw_text(display, label, (4, 4), (0, 255, 0), 14)

        self._show(display)

    # ── TRYB 2: pełne zdjęcie — segmentacja kolor + kształt ──────
    def _detection_mode(self, img):
        self.mode_label.config(text="Tryb: detekcja (segmentacja kolor + kształt)")

        MAX_WIDTH = 900
        if img.shape[1] > MAX_WIDTH:
            r = MAX_WIDTH / img.shape[1]
            img = cv2.resize(img, (MAX_WIDTH, int(img.shape[0] * r)))

        clone = img.copy()
        img_area = img.shape[0] * img.shape[1]

        mask = build_color_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes, confidences, class_ids = [], [], []      # znaki docelowe
        rejected_boxes = []                             # kandydaci-znaki spoza projektu

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA_RATIO * img_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if not (0.6 < aspect < 1.7):   # znaki ~kwadratowe / rombowe
                continue

            # Aproksymacja kształtu — znak to wielokąt foremny / koło
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 3:
                continue

            # Wytnij kandydata z marginesem
            pad = int(0.08 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0 or color_ratio(roi) < MIN_COLOR_RATIO:
                continue

            pred, conf = self._predict(roi)

            # Kandydat wygląda jak znak (kształt + kolor), ale to nie nasza klasa
            if pred == REJECT_CLASS or conf < CONFIDENCE_THRESHOLD:
                rejected_boxes.append([x1, y1, x2 - x1, y2 - y1])
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            class_ids.append(pred)

        # ── Rysowanie znaków docelowych (zielone) z NMS ───────────
        found = []
        if boxes:
            confs = np.array(confidences, dtype=np.float32)
            if confs.min() < 0:
                confs -= confs.min()
            indices = cv2.dnn.NMSBoxes(boxes, confs.tolist(),
                                       score_threshold=0.0, nms_threshold=0.3)
            for i in np.array(indices).flatten():
                x, y, w, h = boxes[i]
                label = CLASS_LABELS[class_ids[i]]
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                clone = self._draw_text(clone, label, (x, max(y - 18, 2)), (0, 255, 0), 13)
                found.append(label)

        # ── Rysowanie znaków spoza projektu (pomarańczowe) z NMS ──
        rejected_count = 0
        if rejected_boxes:
            dummy = [1.0] * len(rejected_boxes)
            r_idx = cv2.dnn.NMSBoxes(rejected_boxes, dummy,
                                     score_threshold=0.0, nms_threshold=0.3)
            for i in np.array(r_idx).flatten():
                x, y, w, h = rejected_boxes[i]
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 140, 255), 2)
                clone = self._draw_text(clone, "Nierozpoznany",
                                        (x, max(y - 18, 2)), (0, 140, 255), 13)
                rejected_count += 1

        # ── Komunikat ─────────────────────────────────────────────
        if found:
            txt = f"Znaleziono: {', '.join(sorted(set(found)))}"
            if rejected_count:
                txt += f"  |  nierozpoznane (spoza projektu): {rejected_count}"
            self.result_label.config(text=txt, fg="#2ecc71")
        elif rejected_count:
            self.result_label.config(
                text=f"Nierozpoznany — wykryto {rejected_count} znak(ów) spoza projektu "
                     "(brak obsługiwanych klas).", fg="#f39c12")
        else:
            self.result_label.config(
                text="Nie wykryto żadnego znaku na zdjęciu.", fg="#e74c3c")

        self._show(clone)

    # ──────────────────────────────────────────────────────────────
    # Rysowanie tekstu (Unicode / polskie znaki) + wyświetlanie
    # ──────────────────────────────────────────────────────────────

    def _draw_text(self, bgr_img, text, xy, color_bgr, font_size=14):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox(xy, text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0))
        draw.text(xy, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _show(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((820, 540))
        tk_img = ImageTk.PhotoImage(pil)
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
