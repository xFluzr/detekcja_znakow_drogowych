# Detekcja i klasyfikacja znaków drogowych (Zadanie 8)

Projekt realizowany w ramach zajęć akademickich, mający na celu stworzenie systemu komputerowego widzenia zdolnego do lokalizacji oraz klasyfikacji znaków drogowych.

## 👥 Zespół Projektowy
* **Jakub Jaszcz** (Lider) – Organizacja repozytorium, przygotowanie bazy danych (GTSRB).
* **Karol Malinowski** – Przetwarzanie obrazów (segmentacja barw HSV, thresholding).
* **Jakub Śliwa** – Przetwarzanie obrazów (transformata Hougha, ekstrakcja cech HOG).
* **Filip Konfederak** – Uczenie Maszynowe (trening klasyfikatora SVM, ewaluacja).
* **Krystian Nowak** – Modele głębokie (YOLOv8-nano, metryki porównawcze, raport).

## 🎯 Opis projektu
Głównym założeniem projektu jest detekcja i klasyfikacja 5 wybranych klas znaków drogowych z bazy **GTSRB (German Traffic Sign Recognition Benchmark)**:
1. Stop
2. Ustąp pierwszeństwa
3. Ograniczenie prędkości
4. Zakaz wjazdu
5. Zakaz ruchu w obu kierunkach

System bada i porównuje dwa podejścia technologiczne:
* **Klasyczny potok wizyjny:** Segmentacja kolorów w przestrzeni HSV $\rightarrow$ Detekcja kształtów (Hough) $\rightarrow$ Ekstrakcja cech (HOG) $\rightarrow$ Klasyfikacja (SVM).
* **Podejście oparte na uczeniu głębokim:** Wykorzystanie pretrenowanego modelu YOLOv8-nano jako nowoczesnego punktu odniesienia (baseline).

---

## ⚙️ Wymagania i instalacja

Projekt został napisany w języku **Python 3.10+**. Aby zainstalować wszystkie niezbędne zależności, sklonuj repozytorium i użyj menedżera pakietów `pip`:

```bash
# 1. Pobranie repozytorium
git clone [https://github.com/xFluazr/detekcja_znakow_drogowych.git](https://github.com/xFluazr/detekcja_znakow_drogowych.git)
cd detekcja_znakow_drogowych

# 2. Utworzenie wirtualnego środowiska (opcjonalnie, ale zalecane)
python -m venv venv
source venv/bin/activate  # na systemach Linux/Mac
venv\Scripts\activate     # na systemach Windows

# 3. Instalacja bibliotek
pip install -r requirements.txt
