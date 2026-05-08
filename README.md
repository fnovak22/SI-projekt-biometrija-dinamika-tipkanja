# MyNotes — privatne bilješke s provjerom dinamike tipkanja

MyNotes je Flask prototip web aplikacije za privatne bilješke. Aplikacija izgleda kao stvarni notes sustav, a dinamika tipkanja koristi se kao dodatni sigurnosni sloj i izvor podataka za ML istraživanje.

## Što je dodano

- hrvatski UI i naziv aplikacije **MyNotes**
- privatne bilješke: kreiranje, uređivanje i brisanje
- drugi korak prijave sada bira **random frazu** iz enrollment skupa, umjesto uvijek prve fraze
- enrollment i login provjera označavaju krivo napisane znakove crveno
- Enter sprema enrollment uzorak / pokreće login provjeru
- uklonjen kratki lažni prikaz greške nakon spremanja uzorka
- free-text uzorci iz editora bilješki spremaju se kao `free_text_note`
- research admin dashboard na `/admin`
- CSV export za ML analizu
- demo research admin korisnik automatski postoji u bazi: `admin` / `admin`
- fixed-text model više ne koristi `backspace_count` / `backspace_ratio`, ali se backspace i dalje sprema u feature extraction i CSV za kasnije free-text eksperimente

## Admin račun

Demo research admin račun je već dodan u bazu:

```text
korisničko ime: admin
lozinka: admin
```

Admin se prijavljuje bez dodatne provjere dinamike tipkanja jer služi za research dashboard, statistike i CSV export.

## Pokretanje

```bash
cd software
python app.py
```

Aplikacija koristi SQLite bazu u `instance/app.db`.

## Napomena za fixed-text toleranciju i debug

Fixed-text provjera koristi One-Class SVM s `nu=0.15`. Prihvaćanje nije ručno postavljen postotak tolerancije, nego model prihvaća pokušaj kada je `prediction == 1`, odnosno kada je `decision_function` score barem `0`.

Backspace se i dalje sprema u `features_json` i CSV export, ali se ne koristi u fixed-text vektoru. Fixed-text vektor trenutno ima redoslijed:

1. `avg_dwell_ms`
2. `avg_dd_interval_ms`
3. `std_dwell_ms`
4. `std_dd_interval_ms`
5. `typing_speed_chars_per_sec`
6. `pause_ratio`

Na stranici drugog koraka prijave browser console ispisuje profil korisnika i detalje svakog pokušaja. Uspješni verify pokušaji se spremaju kao `verify_success` i koriste se kod sljedećih treniranja modela. Neuspješni pokušaji se spremaju kao `verify_failed` i ne koriste se za treniranje.
