import numpy as np


def _safe_list(value):
    return value if isinstance(value, list) else []


def _std(values):
    values = _safe_list(values)
    return round(float(np.std(values)), 3) if values else 0.0


def _avg(values):
    values = _safe_list(values)
    return round(float(np.mean(values)), 3) if values else 0.0


def features_to_vector_fixed(features):
    dwell_times = _safe_list(features.get("dwell_times_ms"))
    dd_intervals = _safe_list(features.get("dd_intervals_ms"))

    duration_ms = features.get("duration_ms") or 1
    keydown_count = features.get("keydown_count") or 1
    char_count = features.get("char_count") or 1
    # Backspace i dalje ostaje spremljen u features_json i CSV exportu,
    # ali ga fixed-text model namjerno ne koristi jer je previše varijabilan.

    avg_dwell = features.get("avg_dwell_ms") or _avg(dwell_times)
    avg_dd = features.get("avg_dd_interval_ms") or _avg(dd_intervals)

    std_dwell = _std(dwell_times)
    std_dd = _std(dd_intervals)

    typing_speed = round((char_count / duration_ms) * 1000, 3)
    long_pause_count = sum(1 for x in dd_intervals if x > 1000)
    pause_ratio = round(long_pause_count / len(dd_intervals), 3) if dd_intervals else 0.0

    return [
        avg_dwell,
        avg_dd,
        std_dwell,
        std_dd,
        typing_speed,
        pause_ratio
    ]