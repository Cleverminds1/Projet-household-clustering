"""
Correctif n°1 — Détection des journées "plates" (capteur figé / valeurs imputées)
===================================================================================
À insérer dans src/features.py, juste après la définition de compute_daily_features().
Modifie build_daily_features_for_house() pour exclure ces journées plutôt que de les
laisser polluer le clustering (voir point de vigilance n°1 du document de transférabilité).

Usage dans le Notebook 02 (Feature Engineering REFIT) : aucun changement de code demandé
à l'utilisateur -- juste remplacer la version de build_daily_features_for_house() par
celle-ci (ou importer flag_flat_days ci-dessous et filtrer après coup).
"""
import numpy as np
import pandas as pd

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


def is_flat_day(power_kw_day: pd.Series, max_run=MAX_FLAT_RUN_MINUTES) -> bool:
    """True si la journée contient un segment de capteur figé plus long que max_run minutes."""
    return longest_constant_run(power_kw_day.values) > max_run


# --- Version corrigée de build_daily_features_for_house (src/features.py) ---
def build_daily_features_for_house_v2(power_kw_minute: pd.Series, house_id: str):
    from features import compute_daily_features, FEATURES_FINAL  # import existant

    rows, flagged = [], 0
    for day_ts, day_data in power_kw_minute.groupby(power_kw_minute.index.normalize()):
        if is_flat_day(day_data):
            flagged += 1
            continue  # jour exclu : capteur figé probable, pas un régime réel
        feat = compute_daily_features(day_data, day_ts)
        if feat is not None:
            feat["house_id"] = house_id
            rows.append(feat)

    print(f"  {house_id} : {flagged} journée(s) plate(s) exclue(s)")
    if not rows:
        return pd.DataFrame(columns=["house_id", "date"] + FEATURES_FINAL)
    df = pd.DataFrame(rows)
    return df[["house_id", "date"] + FEATURES_FINAL]


# Après remplacement : relancer Notebook 02 (features), puis 03 (clustering Niveau 1).
# Comparer k, Silhouette et poids d'is_weekend avant/après -- rapporter le nombre de
# journées exclues par foyer dans l'article (matériel supplémentaire).
