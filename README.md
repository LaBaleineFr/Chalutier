# Chalutier
Outil d'aide à la décision d'optimisation de portefeuille de cryptomonnaies

## Endpoint

POST
``` http://localhost:5000/optimise ```

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
