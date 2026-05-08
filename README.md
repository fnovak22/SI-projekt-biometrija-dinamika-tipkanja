# KeyStroke ID — prototip za dinamiku tipkanja

Tema projekta: **Identifikacija korisnika pomoću dinamike tipkanja**.

Ova verzija prototipa ima:

- registraciju koja se završava tek nakon **20 obaveznih enrollment uzoraka**
- 5 fiksnih duljih fraza ponovljenih 4 puta (`5 x 4 = 20`)
- dvo-korak prijavu: prvo lozinka, zatim provjera tipkanja
- placeholder za SVM model dok ML tim ne spoji stvarni model
- spremanje raw `keydown`/`keyup` eventova i osnovnih featurea u SQLite
- CSV export za ML tim

## Pokretanje

```bash
python -m venv venv

(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& C:\SI-projekt-biometrija-dinamika-tipkanja\venv\Scripts\Activate.ps1)

pip install -r requirements.txt
python software/app.py
```

Otvori:

```text
http://127.0.0.1:5000
```

## Testiranje aplikacije

1. Otvori `/register`.
2. Unesi korisničko ime i lozinku.
3. Nakon toga prepiši svih 20 fraza koje aplikacija traži.
4. Tek nakon 20 spremljenih uzoraka registracija je završena.
5. Prijavi se na `/login`.
6. Nakon točne lozinke aplikacija traži dodatnu frazu za provjeru tipkanja.
7. Budući da model još nije spojen, taj korak trenutno samo sprema uzorak kao `verify_attempt` i propušta korisnika ako je tekst točno prepisan.

## Gdje su podaci?

SQLite baza:

```text
instance/app.db
```

Tablica:

```text
typing_sample
```

CSV export:

```text
/export/typing-samples.csv
```

Važni stupci za ML tim:

- `username` — labela korisnika
- `sample_type` — `enroll`, `extra_enroll` ili `verify_attempt`
- `prompt_id` — koja je fraza pisana
- `prompt_text`
- `typed_text`
- `duration_ms`
- `backspace_count`
- `avg_dwell_ms`
- `avg_dd_interval_ms`
- `dwell_times_ms`
- `dd_intervals_ms`
- `events_json`

Za prvi model preporuka je krenuti s `sample_type=enroll`, grupirati po `username` i `prompt_id`, te koristiti `dwell_times_ms` i `dd_intervals_ms`.

## Gdje se spaja model?

U `software/app.py` postoji funkcija:

```python
verify_typing_with_model_stub(user, features, prompt_id)
```

To je placeholder. ML tim kasnije tamo može ubaciti poziv na istrenirani SVM model.
