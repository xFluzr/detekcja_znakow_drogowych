# Wizja Komputerowa - Detekcja i Klasyfikacja Znaków Drogowych

**Autorzy:** Jakub Jaszcz, Karol Malinowski, Jakub Śliwa, Filip Konfederak, Krystian Nowak

## Krótki opis projektu
Projekt ten realizuje system wizyjny służący do detekcji oraz klasyfikacji znaków drogowych w oparciu o klasyczne metody wizji komputerowej oraz uczenie maszynowe. Wykorzystuje bazę The German Traffic Sign Recognition Benchmark (GTSRB). W projekcie użyto dwóch różnych podejść do segmentacji znaków (klasyczne kontury vs MSER) wraz z weryfikacją w przestrzeni barw HSV. Następnie na wyodrębnionych obiektach dokonano ekstrakcji cech przy pomocy deskryptora HOG (Histogram of Oriented Gradients) i przeprowadzono ich klasyfikację maszyną wektorów nośnych (SVM) z jądrem liniowym. Projekt zawiera również graficzną aplikację desktopową (GUI) ułatwiającą testowanie modelu.

## Wymagania wstępne
Projekt wymaga języka Python w wersji 3.8 lub nowszej. Aby zainstalować wymagane biblioteki, wykonaj w terminalu polecenie:
```bash
pip install -r requirements.txt
```

Przed uruchomieniem skryptów, upewnij się, że w głównym katalogu projektu znajduje się folder `archive/`, który zawiera rozpakowany zbiór danych GTSRB (w tym pliki `Train.csv`, `Test.csv` oraz foldery ze zdjęciami).

## Instrukcja uruchomienia (po kolei)

### Krok 1: Wstępne przetwarzanie i filtracja bazy
Skrypt ekstrahuje z plików CSV tylko 5 interesujących nas klas, kadruje je po współrzędnych ROI i zmniejsza do rozmiaru 64x64.
```bash
python database-filter.py
```
*Po wykonaniu tego kroku utworzony zostanie folder `processed_data/` z danymi treningowymi i testowymi.*

### Krok 2: Ekstrakcja cech i Klasyfikacja (Trening SVM)
Główny potok wczytujący wyfiltrowane zdjęcia, obliczający wektory HOG i przeprowadzający klasyfikację modelem SVM. Skrypt testuje też model na zbiorze testowym, wypisuje metryki oraz zapisuje ewentualne pomyłki modelu. Na końcu tworzy plik **`svm_model.joblib`**.
```bash
python svm_training_and_prediction.py
```
*Błędnie sklasyfikowane znaki drogowe zostaną skopiowane do folderu `processed_data/errors/` z dokładnym opisem w nazwie pliku.*

### Krok 3: Uruchomienie aplikacji graficznej (Desktop GUI)
Kiedy posiadasz już wytrenowany model `svm_model.joblib`, możesz odpalić aplikację okienkową, która pozwala na ręczne wybieranie zdjęć znaków i ich rozpoznawanie.
```bash
python gui.py
```

### Dodatkowo: Testowanie algorytmów segmentacji (opcjonalne)
Możesz przetestować skuteczność działania poszczególnych algorytmów wykrawających znaki z szerokiego tła (analizując skuteczność krawędzi Canny'ego lub detektora MSER).
```bash
python image-processing.py
# lub
python mser_image_processing.py
```
