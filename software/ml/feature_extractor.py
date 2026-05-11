import numpy as np


# Tipke koje NISU dio fixed-text biometrijskog ritma.
# Raw događaji se i dalje spremaju u bazu, ali fixed-text model iz njih računa
# "čisti" ritam samo za znakove koji ostaju u završnom tekstu.
FIXED_TEXT_IGNORED_KEYS = {
    "Backspace", "Delete",
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "Home", "End", "PageUp", "PageDown",
    "Tab", "Escape", "Enter",
    "Shift", "Control", "Alt", "Meta", "CapsLock",
}

# Pauze dulje od ovog praga ne ulaze direktno u prosjek/std razmaka između
# pritisaka, jer jedna pauza od npr. 20 sekundi može potpuno uništiti prosjek.
# Umjesto toga se takve pauze mjere zasebno kroz pause_ratio.
LONG_PAUSE_THRESHOLD_MS = 1500



# NAPOMENA ZA ML TIM:
# Za free-text biometriju NEMOJTE koristiti fixed-text filter.
# Free-text model treba uključiti cijeli signal iz events_json/features_json, uključujući
# Backspace, Delete, pauze, ispravke, navigacijske tipke i backspace_count.
# Ovaj filter postoji samo zato što je fixed-text provjera osjetljiva na situaciju
# gdje korisnik pogriješi jednu frazu pa spamom Backspacea pokvari vremenske značajke.


def _safe_list(value):
    return value if isinstance(value, list) else []


def _std(values):
    values = _safe_list(values)
    return round(float(np.std(values)), 3) if values else 0.0


def _avg(values):
    values = _safe_list(values)
    return round(float(np.mean(values)), 3) if values else 0.0


def _is_keydown(event):
    return event.get("type") == "keydown" and not event.get("repeat")


def _is_keyup(event):
    return event.get("type") == "keyup"


def _is_printable_character_key(event):
    key = str(event.get("key", ""))
    return (
        len(key) == 1
        and key not in FIXED_TEXT_IGNORED_KEYS
        and not event.get("ctrlKey")
        and not event.get("altKey")
        and not event.get("metaKey")
    )


def _event_time(event):
    try:
        return float(event.get("t", 0))
    except (TypeError, ValueError):
        return 0.0


def _empty_features():
    return {
        "duration_ms": 0.0,
        "raw_duration_ms": 0.0,
        "active_duration_ms": 0.0,
        "long_pause_count": 0,
        "long_pause_threshold_ms": LONG_PAUSE_THRESHOLD_MS,
        "keydown_count": 0,
        "keyup_count": 0,
        "char_count": 0,
        "backspace_count": 0,
        "avg_dwell_ms": None,
        "avg_dd_interval_ms": None,
        "std_dwell_ms": 0.0,
        "std_dd_interval_ms": 0.0,
        "typing_speed_chars_per_sec": 0.0,
        "pause_ratio": 0.0,
        "dwell_times_ms": [],
        "dd_intervals_ms": [],
        "normal_dd_intervals_ms": [],
    }


def _features_from_key_events(keydowns, keyups, all_events=None):
    """Računa osnovne featuree iz zadanih keydown/keyup događaja."""
    all_events = all_events if all_events is not None else (keydowns + keyups)
    if not keydowns and not keyups:
        return _empty_features()

    pending_down = {}
    dwell_times = []
    for event in sorted(all_events, key=_event_time):
        key = event.get("key")
        code = event.get("code")
        t = _event_time(event)
        k = f"{key}|{code}"
        if _is_keydown(event):
            pending_down.setdefault(k, []).append(t)
        elif _is_keyup(event) and pending_down.get(k):
            down_t = pending_down[k].pop(0)
            dwell_times.append(round(t - down_t, 3))

    dd_intervals = []
    sorted_keydowns = sorted(keydowns, key=_event_time)
    for i in range(1, len(sorted_keydowns)):
        dd_intervals.append(round(_event_time(sorted_keydowns[i]) - _event_time(sorted_keydowns[i - 1]), 3))

    typed_chars = [e.get("key") for e in sorted_keydowns if _is_printable_character_key(e)]

    raw_duration_ms = 0.0
    sorted_events = sorted(all_events, key=_event_time)
    if len(sorted_events) >= 2:
        raw_duration_ms = round(_event_time(sorted_events[-1]) - _event_time(sorted_events[0]), 3)

    normal_dd_intervals = [x for x in dd_intervals if x <= LONG_PAUSE_THRESHOLD_MS]
    long_pause_count = sum(1 for x in dd_intervals if x > LONG_PAUSE_THRESHOLD_MS)
    pause_ratio = round(long_pause_count / len(dd_intervals), 3) if dd_intervals else 0.0

    # Za brzinu tipkanja koristimo aktivno trajanje: duge pauze se ne brišu u
    # potpunosti, nego im ostavljamo maksimalno LONG_PAUSE_THRESHOLD_MS. Tako
    # korisnik smije zastati i razmisliti, a model svejedno dobiva informaciju
    # da je pauza postojala kroz pause_ratio.
    removed_pause_ms = sum(max(0.0, x - LONG_PAUSE_THRESHOLD_MS) for x in dd_intervals)
    active_duration_ms = max(0.0, raw_duration_ms - removed_pause_ms)
    active_duration_ms = round(active_duration_ms, 3)

    char_count = len(typed_chars)
    typing_speed = round((char_count / active_duration_ms) * 1000, 3) if active_duration_ms else 0.0

    return {
        "duration_ms": active_duration_ms,
        "raw_duration_ms": raw_duration_ms,
        "active_duration_ms": active_duration_ms,
        "long_pause_count": long_pause_count,
        "long_pause_threshold_ms": LONG_PAUSE_THRESHOLD_MS,
        "keydown_count": len(sorted_keydowns),
        "keyup_count": len(keyups),
        "char_count": char_count,
        "backspace_count": sum(1 for e in sorted_keydowns if e.get("key") == "Backspace"),
        "avg_dwell_ms": _avg(dwell_times) if dwell_times else None,
        "avg_dd_interval_ms": _avg(normal_dd_intervals) if normal_dd_intervals else None,
        "std_dwell_ms": _std(dwell_times),
        "std_dd_interval_ms": _std(normal_dd_intervals),
        "typing_speed_chars_per_sec": typing_speed,
        "pause_ratio": pause_ratio,
        "dwell_times_ms": dwell_times,
        "dd_intervals_ms": dd_intervals,
        "normal_dd_intervals_ms": normal_dd_intervals,
    }


def extract_features(events):
    """Raw feature extraction za opću/free-text analizu. Uključuje sve tipke."""
    events = _safe_list(events)
    keydowns = [e for e in events if _is_keydown(e)]
    keyups = [e for e in events if _is_keyup(e)]
    return _features_from_key_events(keydowns, keyups, events)


def extract_fixed_text_features(events, final_text=None):
    """Feature extraction za fixed-text model.

    Strategija:
    1. Raw events_json ostaje netaknut za istraživanje.
    2. Za fixed-text model ignoriraju se Backspace/Delete/navigation/modifier tipke.
    3. Ako korisnik pogriješi pa obriše znak, simuliramo završni tekst i iz modela
       izbacujemo znakove koji su obrisani. Tako spam Backspacea ne kvari prosječne
       dwell/DD vrijednosti fixed-text provjere.
    4. Free-text model NE smije koristiti ovu funkciju; on treba cijeli signal.
    """
    events = _safe_list(events)

    # Uparivanje keydown/keyup po konkretnom događaju kako bismo mogli sačuvati dwell
    # samo za znakove koji ostaju u završnom tekstu.
    pending = {}
    printable_tokens = []
    event_to_token = {}

    for index, event in enumerate(events):
        key = event.get("key")
        code = event.get("code")
        pair_key = f"{key}|{code}"
        if _is_keydown(event) and _is_printable_character_key(event):
            token = {
                "char": key,
                "keydown": {**event, "_idx": index},
                "keyup": None,
            }
            pending.setdefault(pair_key, []).append(token)
            printable_tokens.append(token)
        elif _is_keyup(event) and pending.get(pair_key):
            token = pending[pair_key].pop(0)
            token["keyup"] = {**event, "_idx": index}
            event_to_token[index] = token

    # Simulacija jednostavnog uređivanja na kraju teksta. Za fixed fraze korisnici
    # u pravilu tipkaju linearno; Backspace uklanja zadnji uneseni znak.
    kept_stack = []
    token_by_keydown_index = {token["keydown"].get("_idx"): token for token in printable_tokens}
    for index, event in enumerate(events):
        if index in token_by_keydown_index:
            kept_stack.append(token_by_keydown_index[index])
        elif _is_keydown(event) and event.get("key") == "Backspace" and kept_stack:
            kept_stack.pop()

    reconstructed = "".join(token["char"] for token in kept_stack)
    if final_text is not None and reconstructed != final_text:
        # Ako rekonstrukcija ne uspije zbog selekcije teksta/kursora, i dalje ostajemo
        # konzervativni: uzimamo sve printable znakove, ali bez edit/navigation tipki.
        kept_stack = printable_tokens

    kept_indices = set()
    kept_keydowns = []
    kept_keyups = []
    for token in kept_stack:
        kd = token.get("keydown")
        ku = token.get("keyup")
        if kd:
            kept_keydowns.append(kd)
            kept_indices.add(kd.get("_idx"))
        if ku:
            kept_keyups.append(ku)
            kept_indices.add(ku.get("_idx"))

    # Za DD intervale želimo smanjiti utjecaj vremena provedenog u obrisanim znakovima
    # i Backspace događajima. Zato računamo komprimirane keydown timestampove: između
    # dva zadržana znaka oduzimamo vremenski raspon ignoriranih događaja.
    compressed_keydowns = []
    removed_time_so_far = 0.0
    previous_kept_idx = None
    previous_kept_t = None
    for kd in kept_keydowns:
        idx = kd.get("_idx")
        t = _event_time(kd)
        if previous_kept_idx is not None:
            ignored_times = [
                _event_time(event)
                for event_index, event in enumerate(events)
                if previous_kept_idx < event_index < idx and event_index not in kept_indices
            ]
            if ignored_times:
                span = max(ignored_times) - min(ignored_times)
                if span > 0:
                    removed_time_so_far += span
        compressed = dict(kd)
        compressed["t"] = round(t - removed_time_so_far, 3)
        compressed_keydowns.append(compressed)
        previous_kept_idx = idx
        previous_kept_t = t

    # Keyup timestampove komprimiramo približno istim offsetom kao pripadni keydown;
    # dwell ostaje stabilan jer se temelji na razlici keyup-keydown za isti znak.
    compressed_events = []
    keydown_offsets = {}
    for original, compressed in zip(kept_keydowns, compressed_keydowns):
        keydown_offsets[original.get("_idx")] = _event_time(original) - _event_time(compressed)
        compressed_events.append(compressed)
    for token in kept_stack:
        kd = token.get("keydown")
        ku = token.get("keyup")
        if kd and ku:
            offset = keydown_offsets.get(kd.get("_idx"), 0.0)
            compressed_ku = dict(ku)
            compressed_ku["t"] = round(_event_time(ku) - offset, 3)
            compressed_events.append(compressed_ku)

    fixed = _features_from_key_events(compressed_keydowns, [e for e in compressed_events if _is_keyup(e)], compressed_events)
    fixed["ignored_keys_for_fixed_text"] = sorted(FIXED_TEXT_IGNORED_KEYS)
    fixed["correction_events_removed_for_fixed_text"] = max(0, len(printable_tokens) - len(kept_stack))
    fixed["reconstructed_text_matches_final"] = final_text is None or reconstructed == final_text
    return fixed


def attach_fixed_text_features(features, events, final_text=None):
    features = dict(features or {})
    features["fixed_text_features"] = extract_fixed_text_features(events, final_text=final_text)
    return features


def _fixed_source(features):
    fixed = features.get("fixed_text_features") if isinstance(features, dict) else None
    return fixed if isinstance(fixed, dict) else features


def features_to_vector_fixed(features):
    # Fixed-text model koristi očišćene fixed_text_features kada postoje.
    # Fallback postoji samo za stare uzorke prije migracije.
    source = _fixed_source(features)
    dwell_times = _safe_list(source.get("dwell_times_ms"))
    dd_intervals = _safe_list(source.get("dd_intervals_ms"))

    duration_ms = source.get("duration_ms") or 1
    char_count = source.get("char_count") or 1

    avg_dwell = source.get("avg_dwell_ms") or _avg(dwell_times)
    avg_dd = source.get("avg_dd_interval_ms") or _avg(dd_intervals)

    std_dwell = source.get("std_dwell_ms") if source.get("std_dwell_ms") is not None else _std(dwell_times)
    std_dd = source.get("std_dd_interval_ms") if source.get("std_dd_interval_ms") is not None else _std(dd_intervals)

    typing_speed = source.get("typing_speed_chars_per_sec")
    if typing_speed is None:
        typing_speed = round((char_count / duration_ms) * 1000, 3)

    pause_ratio = source.get("pause_ratio")
    if pause_ratio is None:
        long_pause_count = sum(1 for x in dd_intervals if x > LONG_PAUSE_THRESHOLD_MS)
        pause_ratio = round(long_pause_count / len(dd_intervals), 3) if dd_intervals else 0.0

    return [
        avg_dwell or 0.0,
        avg_dd or 0.0,
        std_dwell or 0.0,
        std_dd or 0.0,
        typing_speed or 0.0,
        pause_ratio or 0.0,
    ]
    
def _normalized_free_feature_values(features):
    """Vraća free-text featuree koji se koriste za model.

    Funkcija namjerno ponovno računa DD prosjek/std i brzinu iz raw intervala kada
    su dostupni. Time i stariji spremljeni uzorci dobivaju isti tretman dugih
    pauza kao novi uzorci: duga pauza ide u pause_ratio, ali ne razbija prosjek
    razmaka ni brzinu tipkanja.
    """
    features = features or {}
    dd_intervals = _safe_list(features.get("dd_intervals_ms"))
    normal_dd_intervals = [x for x in dd_intervals if x <= LONG_PAUSE_THRESHOLD_MS]

    avg_dd = _avg(normal_dd_intervals) if normal_dd_intervals else features.get("avg_dd_interval_ms", 0.0)
    std_dd = _std(normal_dd_intervals) if normal_dd_intervals else features.get("std_dd_interval_ms", 0.0)

    pause_ratio = features.get("pause_ratio")
    if dd_intervals:
        long_pause_count = sum(1 for x in dd_intervals if x > LONG_PAUSE_THRESHOLD_MS)
        pause_ratio = round(long_pause_count / len(dd_intervals), 3)
    elif pause_ratio is None:
        pause_ratio = 0.0

    duration_ms = features.get("duration_ms", 0.0)
    if dd_intervals:
        raw_duration = features.get("raw_duration_ms", duration_ms) or 0.0
        removed_pause_ms = sum(max(0.0, x - LONG_PAUSE_THRESHOLD_MS) for x in dd_intervals)
        duration_ms = max(0.0, float(raw_duration) - removed_pause_ms)

    char_count = float(features.get("char_count", 0.0))
    typing_speed = features.get("typing_speed_chars_per_sec")
    if dd_intervals:
        typing_speed = round((char_count / duration_ms) * 1000, 3) if duration_ms else 0.0
    elif typing_speed is None:
        typing_speed = 0.0

    backspace_count = float(features.get("backspace_count", 0.0) or 0.0)
    correction_ratio = round(backspace_count / char_count, 3) if char_count else 0.0

    return {
        # Duljina teksta i ukupno trajanje nisu dio free-text vektora jer korisnik
        # u bilješkama svaki put piše drugačije dugačak tekst. Zadržavamo samo
        # ritmičke značajke koje su usporedive između različitih odlomaka.
        "duration_ms": float(duration_ms or 0.0),
        "keydown_count": float(features.get("keydown_count", 0.0)),
        "keyup_count": float(features.get("keyup_count", 0.0)),
        "char_count": char_count,
        "backspace_count": backspace_count,
        "correction_ratio": float(correction_ratio or 0.0),
        "avg_dwell_ms": float(features.get("avg_dwell_ms", 0.0) or 0.0),
        "avg_dd_interval_ms": float(avg_dd or 0.0),
        "std_dwell_ms": float(features.get("std_dwell_ms", 0.0) or 0.0),
        "std_dd_interval_ms": float(std_dd or 0.0),
        "typing_speed_chars_per_sec": float(typing_speed or 0.0),
        "pause_ratio": float(pause_ratio or 0.0),
    }


def features_to_vector_free(features):
    values = _normalized_free_feature_values(features)
    return [
        values["avg_dwell_ms"],
        values["avg_dd_interval_ms"],
        values["std_dwell_ms"],
        values["std_dd_interval_ms"],
        values["typing_speed_chars_per_sec"],
        values["pause_ratio"],
        values["correction_ratio"],
    ]
