"""
Correctif n°2 — Niveau 2 sans is_weekend (point de vigilance n°2)
=================================================================
À l'échelle d'un ménage, is_weekend n'est que la part de week-ends parmi les jours
retenus (~2/7 pour tous) : ce n'est pas un comportement. On l'exclut de la signature.

Dans le Notebook 04, remplacer UNIQUEMENT :
    X_household_raw = df_household[FEATURES_FINAL].values
par :
    X_household_raw = df_household[FEATURES_L2].values
et définir FEATURES_L2 comme ci-dessous. Remplacer aussi FEATURES_FINAL par FEATURES_L2
dans : le bootstrap (bootstrap_stability), le groupby des profils, et le Notebook 05
(explainability Niveau 2 : X_l2_scaled, shap, arbre, feature_names).
"""
from features import FEATURES_FINAL

FEATURES_L2 = [f for f in FEATURES_FINAL if f != "is_weekend"]   # 6 variables
print("Variables Niveau 2 :", FEATURES_L2)

# Dans le Notebook 04 -- les lignes à modifier (4 endroits) :
# 1) construction de la signature :
#       df_household = df_level1_raw.groupby("house_id")[FEATURES_L2].mean().reset_index()
# 2) mise à l'échelle + clustering :
#       X_household_raw = df_household[FEATURES_L2].values
# 3) bootstrap : dans bootstrap_stability(), remplacer FEATURES_FINAL par FEATURES_L2
#       days = df_level1_raw.loc[df_level1_raw["house_id"] == house_id, FEATURES_L2].values
# 4) profils : profile_cols = FEATURES_L2 + ["occupancy", "appliances_owned", "bedrooms", "construction_year"]

# À rapporter dans l'article après ré-exécution :
#   - k retenu, Silhouette, stabilité bootstrap moyenne (min-max)
#   - nouveaux tests de validation externe (Kruskal-Wallis, chi2/V de Cramér)
#   - arbre de décision : vérifier qu'is_weekend n'apparaît plus, et que 3 variables
#     environ suffisent encore (sinon le signaler comme surapprentissage à n=20)
