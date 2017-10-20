# Chalutier

Outil d'aide à la décision d'optimisation de portefeuille de cryptomonnaies

Construit un portefeuille efficient de Markowitz sur les cinq derniers jours.

## Installation
Testé pour python 3.[4,5,6]
```
pip3 install -r requirements.txt
```

## Utilisation

### CLI
```
python3 chalutier-cli.py --currencies LTC SC XVG
```

### Service web
```
python3 chalutier.py
```

Utiliser le paramètre `--port` pour changer le port du service

#### Endpoint


```
POST http://localhost:5000/optimise
```

BODY :
```
{
  "currencies": [
    "LTC",
    "SC",
    "XVG"
  ]
}
```
(2 à 10 indices)

et le résultat :

```
{
  "result":{
    "LTC": 0.835421651,
    "SC": ...
    "XVG": ...
  }
}
```
