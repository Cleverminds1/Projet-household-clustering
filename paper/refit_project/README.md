# Segmentation explicable des ménages — Pipeline REFIT

Migration du pipeline UCI (TFC, foyer unique) vers REFIT (20 foyers, Royaume-Uni) +
UCI (réplication de référence), avec ajout d'un **Niveau 2** de segmentation
inter-foyer validée par métadonnées socio-démographiques réelles.

## Structure du dossier

```
.
├── README.md                          <- ce fichier
├── data/
│   ├── raw/
│   │   ├── refit/                     <- À PEUPLER : house-1.csv ... house-21.csv
│   │   │                                 (house-14 n'existe pas)
│   │   │                                 Télécharger CLEAN_REFIT_081116.7z sur
│   │   │                                 https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/
│   │   ├── CLEAN_READ_ME_081116.txt   <- déjà fourni
│   │   └── MetaData_Tables.xlsx       <- déjà fourni (métadonnées ménages)
│   └── processed/
│       ├── uci/                       <- résultats déjà produits dans le TFC original
│       └── refit/                     <- généré par les notebooks 01 et 02
├── notebooks/
│   ├── 01_EDA_REFIT.ipynb                          Chargement, nettoyage, métadonnées
│   ├── 02_Feature_Engineering_REFIT.ipynb          7 features journalières x 20 foyers
│   ├── 03_Clustering_Level1_Intra_Foyer.ipynb      Réplication du clustering x 20 foyers
│   ├── 04_Clustering_Level2_Household_Typology.ipynb  VRAIE segmentation des ménages (n=20)
│   ├── 05_Explainability.ipynb                     XGBoost + SHAP + arbre, Niveau 1 et 2
│   └── 06_Robustness_Tests.ipynb                   Ablation, bruit, agrégation, durée réduite
├── src/
│   ├── features.py                    <- feature engineering partagé (identique UCI/REFIT)
│   └── clustering_pipeline.py         <- KMedoids, Cycle 1, Cycle 2, sélection finale
├── images/                            <- figures générées par chaque notebook (1 sous-dossier/notebook)
└── results/                           <- tableaux de résultats (CSV) produits par les notebooks
```

## Ordre d'exécution

Les notebooks sont **numérotés dans leur ordre d'exécution obligatoire** — chacun dépend des
fichiers produits par le précédent (`data/processed/refit/*.csv`, `results/*.csv`). Ouvrir et
exécuter dans l'ordre 01 → 06.

## Avant de commencer

1. Télécharger les 20 fichiers `house-N.csv` (voir lien ci-dessus) et les placer dans
   `data/raw/refit/`.
2. Installer les dépendances manquantes si nécessaire :
   ```
   pip install xgboost shap --break-system-packages
   ```
   (numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, openpyxl sont supposés déjà présents)

## Notes de rigueur importantes (à intégrer dans l'article)

1. **Sept features, pas dix.** Un notebook intermédiaire du TFC original construisait 10
   features (incluant `power_factor`, `sub1_energy_pct`, `sub2_energy_pct`), mais le notebook
   qui a réellement produit les résultats publiés (`04_UCI_clustering.pdf`) n'en utilisait que
   7 (conforme au Tab. 13 du mémoire). `src/features.py` suit cette version à 7 features — la
   seule transférable à REFIT.

2. **Valeur de repli de `day_night_ratio`.** Les notebooks originaux divergent (0, 10, ou
   « 999 » selon le document) quand la consommation nocturne est nulle. `src/features.py`
   retient 999 (valeur documentée dans le mémoire), cas de toute façon extrêmement rare.

3. **Sélection finale entre familles d'algorithmes.** Le TFC comparait des scores composites
   renormalisés séparément par algorithme (non comparables entre eux). Le nouveau pipeline
   (`clustering_pipeline.py::select_best_across_algorithms`) compare le Silhouette Score BRUT,
   directement comparable — à signaler comme amélioration méthodologique dans l'article.

4. **Hyperparamètres d'arbre de décision proportionnels à N.** Le TFC original fixait
   `min_samples_leaf=50` / `min_samples_split=100`, calibrés pour n=1431 jours (UCI). Réutilisés
   tels quels, ces seuils rendraient tout arbre de décision au Niveau 2 (n=20 ménages)
   impossible à diviser. `05_Explainability.ipynb` calcule ces hyperparamètres proportionnellement
   à N (fonction `scaled_tree_params`).

5. **n=20 au Niveau 2 reste un échantillon modeste.** Le clustering inter-foyer produit un
   résultat exploratoire, à renforcer par le bootstrap de stabilité (Notebook 04, Étape 5) et à
   présenter avec prudence statistique dans la Discussion de l'article.
