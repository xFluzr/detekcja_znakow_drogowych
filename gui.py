import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import os
import pickle  # Dodane dla bramkarz_model.pkl

# Słownik klas znaków (Zmień jeśli masz inne)
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


# --- NOWE FUNKCJE GEOMETRYCZNE ---
def sliding_window(image, step_size, window_size):
    """Przesuwa okno po obrazie."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def image_pyramid(image, scale=1.5, min_size=(64, 64)):
    """Tworzy piramidę obrazów (zmniejsza obraz w pętli)."""
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Znaków - Sliding Window + Bramkarz")
        self.root.geometry("800x700")  # Powiększyłem okno
        self.root.configure(bg="#2b2b2b")

        self.hog = get_hog_descriptor()
        self.expert_model = None
        self.gatekeeper_model = None

        self.load_models()
        self.setup_ui()

    def load_models(self):
        # Wczytanie Eksperta
        if os.path.exists('svm_model.joblib'):
            self.expert_model = joblib.load('svm_model.joblib')
        else:
            messagebox.showwarning("Brak modelu", "Brak 'svm_model.joblib'.")

        # Wczytanie Bramkarza
        if os.path.exists('bramkarz_model.pkl'):
            try:
                # ZMIANA TUTAJ: Używamy joblib zamiast pickle
                self.gatekeeper_model = joblib.load('bramkarz_model.pkl')
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować bramkarza: {e}")
        else:
            messagebox.showwarning("Brak modelu", "Brak 'bramkarz_model.pkl'.")

    def setup_ui(self):
        title_label = tk.Label(self.root, text="Wgraj pełne zdjęcie (np. z ulicy)", font=("Helvetica", 16, "bold"),
                               bg="#2b2b2b", fg="white")
        title_label.pack(pady=10)

        btn_select = tk.Button(self.root, text="Wybierz zdjęcie i Skanuj", font=("Helvetica", 12), bg="#3b82f6",
                               fg="white", command=self.open_file)
        btn_select.pack(pady=5)

        self.result_label = tk.Label(self.root, text="Oczekuję na zdjęcie...", font=("Helvetica", 14), bg="#2b2b2b",
                                     fg="#94a3b8")
        self.result_label.pack(pady=10)

        self.image_panel = tk.Label(self.root, bg="#2b2b2b")
        self.image_panel.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.predict_sign(file_path)

    def predict_sign(self, file_path):
        if not self.expert_model or not self.gatekeeper_model:
            self.result_label.config(text="Błąd: Modele nie są załadowane!", fg="#ef4444")
            return

        self.result_label.config(text="Skanowanie obrazu (to może chwilę potrwać)...", fg="#f1c40f")
        self.root.update()

        # Wczytanie obrazka
        stream = open(file_path, "rb")
        bytes_img = bytearray(stream.read())
        numpyarray = np.asarray(bytes_img, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

        # Zmniejszamy obraz początkowy, żeby skanowanie nie trwało 10 minut
        # Możesz to wyłączyć, jeśli chcesz testować na pełnych 4K
        max_width = 800
        if img.shape[1] > max_width:
            ratio = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * ratio)))

        clone = img.copy()

        # Parametry skanowania
        window_size = (64, 64)
        step_size = 16  # Co ile pikseli przesuwać okno. Im mniej, tym wolniej, ale dokładniej.
        scale = 1.3  # Współczynnik zmniejszania w piramidzie

        boxes = []
        confidences = []
        class_ids = []

        current_scale = 1.0

        # ETAP 1: Skanowanie całej piramidy
        for resized in image_pyramid(img, scale=scale, min_size=window_size):
            for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=window_size):
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    continue

                # Liczymy HOG dla okienka
                hog_vector = self.hog.compute(window).flatten()

                # PYTAMY BRAMKARZA
                # Ustaw tutaj próg odcięcia.
                # Wartość 0.85 oznacza: "Przepuść tylko, jeśli masz 85% pewności"
                threshold_proba = 0.65

                # Zmienna dla SVM (jeśli nie używałeś prawdopodobieństw podczas treningu)
                # Standardowy próg to 0.0. Podniesienie do np. 0.5 lub 1.0 wyeliminuje słabe trafienia.
                threshold_decision = 0.8

                is_sign_passed = False

                try:
                    # PODEJŚCIE 1: Dla modeli wspierających prawdopodobieństwo
                    # np. RandomForest, LogisticRegression, lub SVM z włączonym probability=True
                    proba = self.gatekeeper_model.predict_proba([hog_vector])[0]

                    # Szukamy klasy z największym prawdopodobieństwem
                    predicted_class_index = np.argmax(proba)
                    predicted_class_label = self.gatekeeper_model.classes_[predicted_class_index]
                    max_prob = proba[predicted_class_index]

                    # Jeśli model twierdzi, że to klasa "1" (Znak) i jest tego pewien w X%
                    if str(predicted_class_label) == "1" and max_prob >= threshold_proba:
                        is_sign_passed = True

                except AttributeError:
                    # PODEJŚCIE 2: Awaryjne, dla modeli typu klasyczny SVM Linear (bez probabilistyki)
                    # Zwraca wartość liczbową (dodatnią dla klasy pozytywnej, ujemną dla negatywnej)
                    decision_val = self.gatekeeper_model.decision_function([hog_vector])[0]

                    # Zwiększamy próg ufności (domyślnie przepuszcza wszystko > 0)
                    if decision_val > threshold_decision:
                        # W tym podejściu zakładamy, że wartości dodatnie to klasa '1' (Znak)
                        # Jeśli masz odwrotnie skonstruowane klasy, zmień na < -threshold_decision
                        is_sign_passed = True

                # Jeśli Bramkarz uznał, że okienko przekroczyło wybrany próg
                if is_sign_passed:

                    # PYTAMY EKSPERTA
                    predicted_class = self.expert_model.predict([hog_vector])[0]

                    # Odtwarzamy współrzędne dla oryginalnego (dużego) obrazka
                    orig_x = int(x * current_scale)
                    orig_y = int(y * current_scale)
                    orig_w = int(window_size[0] * current_scale)
                    orig_h = int(window_size[1] * current_scale)

                    # Dodajemy do listy potencjalnych znaków
                    boxes.append([orig_x, orig_y, orig_w, orig_h])

                    # Skoro mamy model SVM z jądrem liniowym, możemy wyciągnąć odległość od marginesu
                    # Zastąpi to "pewność" (confidence) potrzebną do odrzucania duplikatów
                    try:
                        conf = float(np.max(self.expert_model.decision_function([hog_vector])))
                    except:
                        conf = 1.0  # Awaryjnie, jeśli np. bramkarz to inny typ modelu

                    confidences.append(float(conf))
                    class_ids.append(int(predicted_class))

            current_scale *= scale

        # ETAP 2: Usuwanie duplikatów (NMS)
        if len(boxes) > 0:
            # score_threshold=0.0 odrzuca absurdalnie słabe wyniki, nms_threshold=0.3 zostawia tylko 1 ramkę dla nakładających się
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.0, nms_threshold=0.3)

            if len(indices) > 0:
                found_signs = []
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    label = CLASS_LABELS.get(class_id, "Nieznany")

                    # Rysujemy zieloną ramkę na zdjęciu
                    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Dodajemy tekst nad ramką
                    cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    found_signs.append(label)

                self.result_label.config(text=f"Znaleziono znaki: {', '.join(set(found_signs))}", fg="#2ecc71")
            else:
                self.result_label.config(text="Bramkarz coś znalazł, ale NMS to odrzucił.", fg="#f39c12")
        else:
            self.result_label.config(text="Nie wykryto żadnych znaków na zdjęciu.", fg="#e74c3c")

        # ETAP 3: Wyświetlanie wyniku w GUI
        clone_rgb = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(clone_rgb)
        img_pil.thumbnail((700, 500))  # Dopasowanie do okna aplikacji
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_panel.configure(image=img_tk)
        self.image_panel.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()