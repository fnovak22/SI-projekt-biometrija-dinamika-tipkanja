# Identificiranje korisnika pomoću dinamike tipkanja

Kratki opis teme: Potrebno je analizirati, modelirati i istražiti kako implementirati metode umjetne inteligencije za analizu dinamike tipkanja kao biometrijske metode za sigurnu autentifikaciju korisnika.

# Instalacija i pokretanje

## 1. Instalacija paketa
Instalirajte najnoviji Python (trenutno 3.14.3): https://www.python.org/downloads/

Potrebno se pozicionirati u korijenski direktorij projekta.

Zatim:
`
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

## 2. Visual Studio Code podešavanje

Da se ne javljaju pogreške kod programiranja u Visual Studio Code, mora se napraviti slijedeće:

Unutar Visual Studio Code dok je otvoren projekt:
 - Ctrl+Shift+P --> "Python: Select Interpreter" --> Python 3.14.3 (venv) .\venv\Scripts\python.exe (Recommended)

## 3. pokretanje programa
python software/app.py


## 4. Pregled baze podataka

Potrebno je instalirati Visual Studio Code ekstenziju "SQLite Viewer" (Florian Klampfe).