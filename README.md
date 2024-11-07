# Návod pro použití knihoven

## Knihovna `my_libs`

### Modul `img`
Obsahuje funkce pro práci s obrázky.

## import do projektu

```python

import sys
sys.path.append('../')
sys.path.append('../my_libs/')
sys.path.append('../my_libs/img/')

import my_libs as ml
```

Jelikož nejde o standardní knihovnu, je nutné ji přidat do cesty. V příkladu je použita cesta `../my_libs/`, která je relativní k aktuálnímu adresáři. Pokud je knihovna v jiném adresáři, je nutné upravit cestu.