"""
src/features.py
================
Feature engineering partagé — IDENTIQUE à la méthodologie validée dans le TFC
(notebooks 02_EDA / feature_engineering / 04_UCI_clustering sur le foyer UCI).

Ce module est importé par tous les notebooks REFIT (01 à 06) afin que le
MÊME code produise les features pour le foyer UCI (déjà fait dans le TFC)
et pour les 20 foyers REFIT. C'est ce qui permet d'affirmer dans l'article
que le pipeline n'a pas été modifié pour s'adapter aux nouvelles données.

NOTE DE RIGUEUR (à lire avant utilisation) :
---------------------------------------------
En comparant les 3 notebooks originaux du TFC (feature_engineering.html,
04_UCI_clustering.pdf, 06_Data_Degradation_Simulation.html), deux
incohérences mineures ont été repérées et corrigées ICI de façon unique :

1. Le notebook `feature_engineering.html` construit et sauvegarde 10
   features (incluant power_factor, sub1_energy_pct, sub2_energy_pct),
   mais le notebook `04_UCI_clustering.pdf` qui produit RÉELLEMENT les
   résultats du TFC charge un fichier `uci_features_scaled.csv` ne
   contenant QUE 7 features (sans power_factor ni sub-metering), conforme
   au tableau 13 du mémoire ("Features éliminées pour non-transférabilité").
   -> Ce module suit la version à 7 features, qui est la version
      RÉELLEMENT utilisée pour produire les résultats publiés, et qui est
      de toute façon la seule transférable à REFIT (pas de sous-compteurs
      cuisine/buanderie comparables dans REFIT, pas de puissance réactive
      fiable pour tous les foyers).

2. La valeur de repli de `day_night_ratio` quand la consommation nocturne
   est nulle diffère selon les notebooks originaux (0 dans
   feature_engineering.html, 10 dans 06_Data_Degradation_Simulation.html,
   "999 (plafond)" dans le texte du mémoire, tableau "Gestion des cas
   limites"). Ce module retient 999, conformément à la version documentée
   dans le mémoire (TFC.pdf), pour disposer d'une règle unique et
   explicite. Ce cas est de toute façon extrêmement rare (aucune
   consommation nocturne sur 8h consécutives).

FEATURES_FINAL est le contrat que tous les notebooks respectent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constantes partagées (identiques au TFC, Tab. 15 "Paramètres et seuils")
# ---------------------------------------------------------------------------

FEATURES_FINAL = [
    "mean_load",        # Tendance centrale (niveau de consommation)
    "load_cv",          # Dispersion relative (stabilité)
    "load_factor",      # Forme de charge (plat vs pics)
    "day_night_ratio",  # Comportement jour/nuit
    "max_load",         # Pic de consommation
    "num_peaks",        # Complexité du profil
    "is_weekend",       # Contexte temporel
]

MINUTES_PER_DAY_EXPECTED = 1440
MIN_MINUTES_PER_DAY = 1380          # Seuil de "jour complet" (>= 90% de 1440), identique au TFC
DAY_HOUR_START, DAY_HOUR_END = 6, 22  # Heures de jour [6h, 22h[ (Tab. 15 du TFC)
DAY_NIGHT_RATIO_CAP = 999.0          # Cas limite documenté (Tab. 16 du TFC) : nuit=0 -> ratio plafonné
PEAK_HEIGHT_MODE = "mean"           # find_peaks(height=hourly.mean(), distance=2), comme dans le TFC
PEAK_DISTANCE_HOURS = 2


def compute_daily_features(power_kw: pd.Series, day_date) -> dict | None:
    """
    Calcule les 7 features journalières à partir d'une série de puissance
    active (en kW), à résolution 1 minute, pour UNE journée calendaire.

    Reproduit fidèlement la logique validée dans le TFC (04_UCI_clustering.pdf) :
    - mean_load, load_cv, load_factor, max_load : calculés directement sur
      la série minute (pas sur une moyenne horaire).
    - num_peaks : calculé sur la série moyennée par heure (24 points),
      via scipy.signal.find_peaks(height=moyenne_horaire, distance=2).
    - day_night_ratio : somme(6h-22h) / somme(22h-6h).
    - is_weekend : dérivé du calendrier (samedi/dimanche), valable en
      France comme au Royaume-Uni.

    Paramètres
    ----------
    power_kw : pd.Series
        Série de puissance active en kW, indexée par un DatetimeIndex à la
        résolution 1 minute, couvrant idéalement une seule journée
        calendaire (jusqu'à 1440 points).
    day_date : date-like
        Date de la journée traitée (utilisée uniquement pour is_weekend).

    Retourne
    --------
    dict | None
        Dictionnaire des 7 features + 'date', ou None si la journée est
        incomplète (< MIN_MINUTES_PER_DAY mesures valides) -- règle
        identique à celle du TFC (exclusion des jours < 1380 minutes).
    """
    gap = power_kw.dropna()
    if gap.shape[0] < MIN_MINUTES_PER_DAY:
        return None

    hour = gap.index.hour

    # --- Tendance centrale et dispersion ---
    mean_load = float(gap.mean())
    std_load = float(gap.std())
    load_cv = std_load / mean_load if mean_load > 0 else 0.0

    # --- Forme de charge ---
    max_load = float(gap.max())
    load_factor = mean_load / max_load if max_load > 0 else 0.0

    day_mask = (hour >= DAY_HOUR_START) & (hour < DAY_HOUR_END)
    night_mask = ~day_mask
    day_consumption = float(gap[day_mask].sum())
    night_consumption = float(gap[night_mask].sum())
    day_night_ratio = (
        day_consumption / night_consumption
        if night_consumption > 0
        else DAY_NIGHT_RATIO_CAP
    )

    # --- Pics (sur la série moyennée par heure, comme dans le TFC) ---
    hourly_gap = gap.groupby(gap.index.hour).mean()
    if len(hourly_gap.dropna()) > 0:
        peaks, _ = find_peaks(
            hourly_gap.values,
            height=hourly_gap.mean(),
            distance=PEAK_DISTANCE_HOURS,
        )
        num_peaks = int(len(peaks))
    else:
        num_peaks = 0

    # --- Contexte temporel ---
    dt = pd.Timestamp(day_date)
    is_weekend = 1 if dt.dayofweek >= 5 else 0  # 5=samedi, 6=dimanche (UK et FR)

    return {
        "date": pd.Timestamp(day_date).date(),
        "mean_load": mean_load,
        "load_cv": load_cv,
        "load_factor": load_factor,
        "day_night_ratio": day_night_ratio,
        "max_load": max_load,
        "num_peaks": num_peaks,
        "is_weekend": is_weekend,
    }


MAX_FLAT_RUN_MINUTES = 60  # seuil : > 1h de valeurs strictement identiques = suspect


def longest_constant_run(values: np.ndarray) -> int:
    """Longueur de la plus longue séquence de valeurs consécutives identiques."""
    if len(values) == 0:
        return 0
    run = max_run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def is_flat_day(power_kw_day: pd.Series, max_run: int = MAX_FLAT_RUN_MINUTES) -> bool:
    """True si la journée contient un segment de capteur figé plus long que max_run minutes."""
    return longest_constant_run(power_kw_day.values) > max_run


def build_daily_features_for_house(
    power_kw_minute: pd.Series, house_id: str
) -> pd.DataFrame:
    """
    Applique compute_daily_features() à chaque journée calendaire d'un
    foyer et retourne un DataFrame (une ligne par jour exploitable).

    Paramètres
    ----------
    power_kw_minute : pd.Series
        Série complète de puissance active (kW) du foyer, résolution
        1 minute, indexée par DatetimeIndex, sur toute la période suivie.
    house_id : str
        Identifiant du foyer (ex. "REFIT_1", "REFIT_21", "UCI_FR").

    Retourne
    --------
    pd.DataFrame
        Colonnes : ['house_id', 'date'] + FEATURES_FINAL.

    NOTE DE PERFORMANCE (corrigée) :
    ---------------------------------
    La version précédente recalculait `power_kw_minute.index.date` (une
    conversion coûteuse de l'index entier) à CHAQUE itération de la
    boucle sur les jours, soit une complexité O(n_jours x n_points) au
    lieu de O(n_points). Sur un foyer REFIT réel (~900 000 points,
    ~640 jours), cela prenait ~70 secondes par appel -- soit ~2h20 pour
    le seul test de bruit du Notebook 06 (20 foyers x 6 niveaux).
    Le regroupement par `power_kw_minute.index.normalize()` ci-dessous
    est strictement équivalent (mêmes résultats, vérifié) et prend
    ~0.5 seconde sur le même foyer (accélération x125).
    """
    rows, flagged = [], 0
    grouped = power_kw_minute.groupby(power_kw_minute.index.normalize())

    for day_ts, day_data in grouped:
        if is_flat_day(day_data):
            flagged += 1
            continue

        feat = compute_daily_features(day_data, day_ts)
        if feat is not None:
            feat["house_id"] = house_id
            rows.append(feat)

    print(f"  {house_id} : {flagged} journée(s) plate(s) exclue(s)")
    if not rows:
        return pd.DataFrame(columns=["house_id", "date"] + FEATURES_FINAL)

    df = pd.DataFrame(rows)
    return df[["house_id", "date"] + FEATURES_FINAL]


# ---------------------------------------------------------------------------
# Fonctions pour les tests de dégradation (agrégation D/W/M) -- Notebook 06
# Généralisation directe de 06_Data_Degradation_Simulation.html à un foyer
# quelconque (identifié par house_id), plutôt qu'au seul foyer UCI.
# ---------------------------------------------------------------------------

def aggregate_to_frequency(power_kw_minute: pd.Series, freq: str) -> pd.DataFrame:
    """
    Agrège une série minute en profils horaires moyens par période
    ('D' jour, 'W' semaine, 'M' mois). Identique à
    06_Data_Degradation_Simulation.html, généralisé à n'importe quelle série.

    Retourne un DataFrame : index = période, colonnes = heures (0-23).
    """
    df = power_kw_minute.to_frame("power_kw")
    df["Hour"] = df.index.hour

    if freq == "D":
        df["Period"] = df.index.date
    elif freq == "W":
        df["Period"] = df.index.to_period("W").start_time
    elif freq == "M":
        df["Period"] = df.index.to_period("M").start_time
    else:
        raise ValueError("freq doit être 'D', 'W' ou 'M'")

    profile_df = df.pivot_table(
        index="Period", columns="Hour", values="power_kw", aggfunc="mean"
    )
    profile_df = profile_df.fillna(profile_df.mean(axis=0))
    return profile_df


def compute_features_from_hourly_profiles(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    À partir d'un DataFrame de profils horaires (index=période,
    colonnes=heures 0-23), calcule les 6 features hors is_weekend
    (identique à compute_7_features_from_profiles du TFC).
    """
    rows = []
    for period in profile_df.index:
        gap = profile_df.loc[period]

        mean_load = gap.mean()
        std_load = gap.std()
        load_cv = std_load / mean_load if mean_load > 0 else 0
        max_load = gap.max()
        load_factor = mean_load / max_load if max_load > 0 else 0

        day_consum = gap.loc[DAY_HOUR_START:DAY_HOUR_END - 1].sum()
        night_hours = list(range(DAY_HOUR_END, 24)) + list(range(0, DAY_HOUR_START))
        night_consum = gap.loc[night_hours].sum()
        day_night_ratio = (
            day_consum / night_consum if night_consum > 0 else DAY_NIGHT_RATIO_CAP
        )

        if len(gap.dropna()) > 0:
            peaks, _ = find_peaks(gap.values, height=gap.mean(), distance=PEAK_DISTANCE_HOURS)
            num_peaks = len(peaks)
        else:
            num_peaks = 0

        rows.append(
            {
                "period": period,
                "mean_load": mean_load,
                "load_cv": load_cv,
                "load_factor": load_factor,
                "day_night_ratio": day_night_ratio,
                "max_load": max_load,
                "num_peaks": num_peaks,
            }
        )
    return pd.DataFrame(rows)


def compute_is_weekend_ratio(power_kw_minute: pd.Series, freq: str) -> pd.DataFrame:
    """
    Calcule la proportion de jours de week-end par période (identique au
    TFC : binaire pour 'D', proportion continue pour 'W'/'M').
    """
    df = power_kw_minute.to_frame("power_kw")
    df["DayOfWeek"] = df.index.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    if freq == "D":
        df["Period"] = df.index.date
    elif freq == "W":
        df["Period"] = df.index.to_period("W").start_time
    elif freq == "M":
        df["Period"] = df.index.to_period("M").start_time
    else:
        raise ValueError("freq doit être 'D', 'W' ou 'M'")

    ratio_df = df.groupby("Period")["IsWeekend"].mean().reset_index()
    ratio_df.columns = ["period", "is_weekend"]
    return ratio_df
