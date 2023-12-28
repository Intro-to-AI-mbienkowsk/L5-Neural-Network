# WSI 2023 LAB5 - Sztuczne sieci neuronowe
### Maksym Bieńkowski

# Zawartość archiwum
## `/src`
* `NeuralNetwork.py` - klasa abstrakcyjna reprezentująca interfejs sieci neuronowej i klasa NoAutogradNeuralNetwork 
implementująca ten interfejs
* `constants.py` - stałe używane w obliczeniach, domyślne argumenty dla parametrów
* `util.py` - funkcje pomocnicze, import danych, wyświetlanie confusion matrix

## Uruchamialne skrypty
### `main.py`
przyjmuje następujące parametry: 
* `-H [rozm1 rozm2 ... rozmN]` - rozmiary ukrytych warstw sieci do wytrenowania, polecam jedną/2 warstwy ukryte nieprzekraczające 100 neuronów, aby trening skończył się w sensownym czasie
* `-E [epochs]` - liczba epok w treningu, 10 wystarcza, a zajmuje mało czasu
* `-L [learning_rate]` - learning rate, polecam wartości rzędu 1-5
* `-T [num_training_examples]` - liczba przykładów treningowych, zalecam wartości między 5000 a 15000.

Wszystkie argumenty mają domyślne wartości, więc możemy uruchomić skrypt poprzez
```shell
python3 -m main
```
lub, określając argumenty:
```shell
python3 -m main -H 50 30 -E 10 -L 3 -T 5000
```
## Krótki opis rozwiązania
Sieć neuronowa wielowarstwowa napisana bez użycia bibliotek AI, służąca do klasyfikacji cyfr ze zbioru MNIST. Po szczegóły implementacyjne zapraszam do raportu.