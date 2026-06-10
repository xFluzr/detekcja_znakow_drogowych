import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import os

# Słownik klas znaków - dopasuj do swoich ID
CLASS_LABELS = {
    2: "Ograniczenie prędkości (50 km/h)",
    12: "Droga z pierwszeństwem",
    14: "Znak STOP",
    17: "Zakaz wjazdu",
    35: "Nakaz jazdy prosto"
}


def get_hog_descriptor():
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def image_pyramid(image, scale=1.5, min_size=(64, 64)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if h < min_size[1] or w < min_size[0]:
            break
        image = cv2.resize(image, (w, h))
        yield image


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Znaków - Sliding Window + Bramkarz")
        self.root.geometry("800x750")
        self.root.configure(bg="#2b2b2b")

        self.hog = get_hog_descriptor()
        self.expert_model = None
        self.gatekeeper_model = None

        self.load_models()
        self.setup_ui()

    def load_models(self):
        if os.path.exists('svm_model.joblib'):
            self.expert_model = joblib.load('svm_model.joblib')

        if os.path.exists('bramkarz_model.pkl'):
            self.gatekeeper_model = joblib.load('bramkarz_model.pkl')
        else:
            messagebox.showwarning("Brak modelu", "Nie znaleziono bramkarz_model.pkl!")

    def setup_ui(self):
        tk.Label(self.root, text="System Rozpoznawania Znaków", font=("Helvetica", 16, "bold"),
                 bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Button(self.root, text="Wybierz zdjęcie i Skanuj", font=("Helvetica", 12), bg="#3b82f6",
                  fg="white", command=self.open_file).pack(pady=5)

        self.result_label = tk.Label(self.root, text="Oczekuję na zdjęcie...", font=("Helvetica", 12),
                                     bg="#2b2b2b", fg="#94a3b8")
        self.result_label.pack(pady=10)

        self.image_panel = tk.Label(self.root, bg="#1e1e1e")
        self.image_panel.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.predict_sign(file_path)

    def predict_sign(self, file_path):
        if not self.expert_model or not self.gatekeeper_model:
            messagebox.showerror("Błąd", "Modele nie zostały załadowane!")
            return

        self.result_label.config(text="Skanowanie w toku...", fg="#f1c40f")
        self.root.update()

        img = cv2.imread(file_path)
        # Skalowanie podglądu dla szybkości
        max_width = 800
        if img.shape[1] > max_width:
            ratio = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * ratio)))

        clone = img.copy()

        # Parametry
        window_size = (64, 64)
        step_size = 16
        scale = 1.3
        threshold_proba = 0.70  # Próg pewności Bramkarza

        boxes, confidences, class_ids = [], [], []
        current_scale = 1.0

        for resized in image_pyramid(img, scale=scale, min_size=window_size):
            for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=window_size):
                hog_vector = self.hog.compute(window).flatten()

                # --- BRAMKARZ (Pipeline z automatycznym skalowaniem) ---
                proba = self.gatekeeper_model.predict_proba([hog_vector])[0]
                pred_class = self.gatekeeper_model.classes_[np.argmax(proba)]
                max_prob = np.max(proba)

                if pred_class == 1 and max_prob >= threshold_proba:
                    # --- EKSPERT ---
                    predicted_class = self.expert_model.predict([hog_vector])[0]

                    orig_x, orig_y = int(x * current_scale), int(y * current_scale)
                    orig_w, orig_h = int(window_size[0] * current_scale), int(window_size[1] * current_scale)

                    boxes.append([orig_x, orig_y, orig_w, orig_h])
                    confidences.append(float(max_prob))
                    class_ids.append(int(predicted_class))

            current_scale *= scale

        # --- NMS ---
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)
            found_signs = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = CLASS_LABELS.get(class_ids[i], "Nieznany")
                    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    found_signs.append(label)
                self.result_label.config(text=f"Znaleziono: {', '.join(set(found_signs))}", fg="#2ecc71")
            else:
                self.result_label.config(text="Bramkarz znalazł, ale Ekspert odrzucił/NMS wyczyścił.", fg="#f39c12")
        else:
            self.result_label.config(text="Nie wykryto żadnego znaku.", fg="#e74c3c")

        # Wyświetlanie
        img_pil = Image.fromarray(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
        img_pil.thumbnail((700, 500))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_panel.configure(image=img_tk)
        self.image_panel.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()