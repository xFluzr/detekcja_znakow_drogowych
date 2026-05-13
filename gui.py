import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import os

# Słownik klas znaków wykorzystanych w projekcie
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

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Znaków Drogowych - SVM")
        self.root.geometry("600x600")
        self.root.configure(bg="#2b2b2b")
        self.root.resizable(False, False)

        # Inicjalizacja deskryptora HOG
        self.hog = get_hog_descriptor()

        # Próba wczytania modelu
        self.model = None
        self.load_model()

        self.setup_ui()

    def load_model(self):
        model_path = 'svm_model.joblib'
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print("Model pomyślnie wczytany.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")
        else:
            messagebox.showwarning(
                "Brak modelu", 
                "Nie znaleziono pliku 'svm_model.joblib'.\nNajpierw wytrenuj model skryptem 'svm_training_and_prediction.py'!"
            )

    def setup_ui(self):
        # Nagłówek
        title_label = tk.Label(
            self.root, 
            text="Wgraj znak drogowy do rozpoznania", 
            font=("Helvetica", 16, "bold"), 
            bg="#2b2b2b", 
            fg="white"
        )
        title_label.pack(pady=20)

        # Przycisk wyboru pliku
        btn_select = tk.Button(
            self.root, 
            text="Wybierz zdjęcie", 
            font=("Helvetica", 12), 
            bg="#3b82f6", 
            fg="white", 
            activebackground="#2563eb",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
            command=self.open_file
        )
        btn_select.pack(pady=10)

        # Miejsce na obrazek
        self.image_panel = tk.Label(self.root, bg="#2b2b2b")
        self.image_panel.pack(pady=20)

        # Etykieta na wynik
        self.result_label = tk.Label(
            self.root, 
            text="Oczekuję na zdjęcie...", 
            font=("Helvetica", 14), 
            bg="#2b2b2b", 
            fg="#94a3b8"
        )
        self.result_label.pack(pady=20)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz zdjęcie znaku",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.ppm")]
        )
        
        if file_path:
            self.display_image(file_path)
            self.predict_sign(file_path)

    def display_image(self, file_path):
        try:
            img = Image.open(file_path)
            # Skalowanie zachowujące proporcje dla celów wyświetlania w GUI
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_panel.configure(image=img_tk)
            self.image_panel.image = img_tk  # Zachowanie referencji zapobiega garbage collection
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wyświetlić obrazu:\n{e}")

    def predict_sign(self, file_path):
        if self.model is None:
            self.result_label.config(text="Błąd: Model SVM nie jest załadowany!", fg="#ef4444")
            return

        self.result_label.config(text="Analizowanie...", fg="#f1c40f")
        self.root.update()

        try:
            # Wczytywanie z polskimi znakami w ścieżkach wymaga użycia numpy
            stream = open(file_path, "rb")
            bytes_img = bytearray(stream.read())
            numpyarray = np.asarray(bytes_img, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Nie można wczytać pliku przez OpenCV.")

            # Normalizacja rozmiaru do okna HOG (64x64)
            img_resized = cv2.resize(img, (64, 64))

            # Ekstrakcja cech
            hog_vector = self.hog.compute(img_resized).flatten()

            # Predykcja
            prediction = self.model.predict([hog_vector])[0]
            predicted_class_id = int(prediction)
            
            label_text = CLASS_LABELS.get(predicted_class_id, f"Nieznana klasa ({predicted_class_id})")

            # Wyświetlenie sukcesu
            self.result_label.config(text=f"Wynik: {label_text}", fg="#2ecc71", font=("Helvetica", 18, "bold"))

        except Exception as e:
            self.result_label.config(text="Błąd analizy", fg="#ef4444")
            messagebox.showerror("Błąd Klasyfikacji", f"Wystąpił błąd podczas predykcji:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
