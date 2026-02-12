# Logs2Graphs

Referencyjna implementacja [DeepLog](https://github.com/Thijsvanede/DeepLog) do wykrywania anomalii w procesie rozruchu systemu operacyjnego.

To repozytorium zawiera referencyjną implementację metody [DeepLog](https://github.com/Thijsvanede/DeepLog), stworzoną na potrzeby pracy inżynierskiej pt. "Wykrywanie anomalii w procesie rozruchu systemu operacyjnego z wykorzystaniem uczenia maszynowego"

## Środowisko

- Python 3.10.14
- libs.txt zawiera pełną listę zależności wymaganych do uruchomienia kodu.

## Zbiory danych

[Link do pobrania zbiorów danych Linux i Windows](https://drive.google.com/drive/folders/1mv--B6TKiTtcy20SNKtbi1xMEW-IuRoE?usp=sharing)

Po pobraniu, odpowiedni plik należy umieścić w katalogu /Data/Windows/Windows.log i analogicznie /Data/Linux/Linux.log

## Uruchomienie

Należy zmienić zmienną `root_path` na początku plików python: `Main.py`, `Parser.py` tak, aby wskazywała odpowiednią ścieżkę do katalogu.

W celu przeprowadzenia eksperymentu i uzyskania wyników dla konkretnego zbioru danych, należy ustawić odpowiednią wartość zmiennej `dataset_name` w pliku `Results.ipynb` i uruchomić wszystkie komórki notatnika.

```python
dataset_name = 'Windows'
dataset_name = 'Linux'
```

## Optymalizacja hiperparametrów

Aby przeprowadzić optymalizację hiperparametrów, należy należy ustawić odpowiednią wartość zmiennej `dataset_name` w pliku `HPO.ipynb` i uruchomić wszystkie komórki notatnika.

```python
dataset_name = 'Windows'
dataset_name = 'Linux'
```

## Materiał źródłowy

```
Thijs van Ede i in. „DeepCASE: Semi-Supervised Contextual Analysis of Security Events”. W:
Proceedings of the IEEE Symposium on Security and Privacy (S&P). IEEE. 2022.
```

```
Min Du i in. „Deeplog: Anomaly detection and diagnosis from system logs through deep learning”.
W: Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017,
s. 1285–1298.
```
