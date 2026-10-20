"""
src/clustering_pipeline.py
===========================
Pipeline de clustering partagé -- reproduit fidèlement 04_UCI_clustering.pdf
(classe KMedoids, fonction evaluate_clustering, protocole Cycle 1 / Cycle 2),
mais encapsulé en fonctions réutilisables afin d'être appelé en boucle sur
21 foyers (Niveau 1) puis une fois sur les 20 signatures ménages (Niveau 2),
plutôt que dupliqué manuellement cellule par cellule comme dans le notebook
original (qui ne traitait qu'un seul foyer).

NOTE DE RIGUEUR (à lire avant utilisation) :
---------------------------------------------
Le TFC original sélectionnait le meilleur modèle FINAL (toutes familles
d'algorithmes confondues) en comparant un "score global" recalculé
séparément (min-max) DANS CHAQUE FAMILLE D'ALGORITHME (K-Means, K-Medoids,
HAC, GMM, DBSCAN chacun normalisé sur son propre balayage de k). Ces scores
ne sont donc pas comparables entre familles d'algorithmes (un score de 0.94
pour un GMM et de 0.74 pour un K-Means ne signifient pas que le premier est
27% meilleur -- ils sont normalisés sur des échelles différentes).

Ce module garde le "score_global" normalisé PAR FAMILLE pour sélectionner
le meilleur k À L'INTÉRIEUR de chaque algorithme (usage légitime), mais la
sélection FINALE entre familles d'algorithmes (fonction
`select_best_across_algorithms`) compare désormais le Silhouette Score
BRUT (non renormalisé), directement comparable puisque calculé sur le même
X à chaque fois. C'est une amélioration délibérée par rapport au TFC
original, à signaler dans la section Méthodes de l'article.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ---------------------------------------------------------------------------
# K-Medoids -- implémentation personnalisée (identique à 04_UCI_clustering.pdf,
# scikit-learn ne fournissant pas de PAM natif)
# ---------------------------------------------------------------------------

class KMedoids:
    """Partitioning Around Medoids (PAM) -- implémentation from-scratch,
    reprise à l'identique du TFC (04_UCI_clustering.pdf, section 0.2)."""

    def __init__(self, n_clusters=8, max_iter=300, random_state=None, metric="euclidean"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric
        self.medoids_indices_ = None
        self.medoids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _compute_distances(self, X, medoids):
        return cdist(X, medoids, metric=self.metric)

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _compute_cost(self, X, labels, medoids_indices):
        total_cost = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                medoid = X[medoids_indices[k]]
                total_cost += np.sum(
                    cdist(cluster_points, medoid.reshape(1, -1), metric=self.metric)
                )
        return total_cost

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        self.medoids_indices_ = rng.choice(n_samples, self.n_clusters, replace=False)
        self.medoids_ = X[self.medoids_indices_].copy()

        distances = self._compute_distances(X, self.medoids_)
        self.labels_ = self._assign_clusters(distances)
        old_cost = self._compute_cost(X, self.labels_, self.medoids_indices_)

        for _ in range(self.max_iter):
            improved = False
            for k in range(self.n_clusters):
                current_medoid_idx = self.medoids_indices_[k]
                cluster_members = np.where(self.labels_ == k)[0]
                if len(cluster_members) == 0:
                    continue

                for candidate_idx in cluster_members:
                    if candidate_idx == current_medoid_idx:
                        continue

                    temp_medoids_indices = self.medoids_indices_.copy()
                    temp_medoids_indices[k] = candidate_idx
                    temp_medoids = X[temp_medoids_indices]
                    temp_distances = self._compute_distances(X, temp_medoids)
                    temp_labels = self._assign_clusters(temp_distances)
                    new_cost = self._compute_cost(X, temp_labels, temp_medoids_indices)

                    if new_cost < old_cost:
                        self.medoids_indices_ = temp_medoids_indices
                        self.medoids_ = X[self.medoids_indices_].copy()
                        self.labels_ = temp_labels
                        old_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        distances = self._compute_distances(X, self.medoids_)
        self.labels_ = self._assign_clusters(distances)
        self.inertia_ = old_cost
        return self

    def predict(self, X):
        distances = self._compute_distances(X, self.medoids_)
        return self._assign_clusters(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
# Évaluation multi-critères (identique à evaluate_clustering du TFC)
# ---------------------------------------------------------------------------

def evaluate_clustering(X, labels, model_name: str = "") -> dict:
    n_clusters = len(np.unique(labels[labels >= 0]))  # exclut le bruit DBSCAN (-1)

    if n_clusters < 2:
        return {
            "model_name": model_name,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": n_clusters,
        }

    return {
        "model_name": model_name,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "n_clusters": n_clusters,
    }


def _min_max_norm(s: pd.Series, invert: bool = False) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0 or np.isnan(rng):
        return pd.Series(0.5, index=s.index)
    norm = (s - s.min()) / rng
    return 1 - norm if invert else norm


# ---------------------------------------------------------------------------
# CYCLE 1 -- K-Means, K-Medoids, HAC-Ward (identique au TFC, section 4.4.2)
# ---------------------------------------------------------------------------

def run_kmeans_sweep(X, k_range=range(2, 11), random_state=42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = model.fit_predict(X)
        res = evaluate_clustering(X, labels, model_name=f"K-Means (k={k})")
        res["k"] = k
        res["inertia"] = model.inertia_
        rows.append(res)
    df = pd.DataFrame(rows)
    df["sil_norm"] = _min_max_norm(df["silhouette"])
    df["db_norm"] = _min_max_norm(df["davies_bouldin"], invert=True)
    df["ch_norm"] = _min_max_norm(df["calinski_harabasz"])
    df["score_global"] = (df["sil_norm"] + df["db_norm"] + df["ch_norm"]) / 3
    return df


def run_kmedoids_sweep(X, k_range=range(2, 7), random_state=42) -> pd.DataFrame:
    # NB : k_range volontairement plus court que K-Means dans le TFC original
    # (K-Medoids est coûteux en O(n^2) par itération) ; ajuster si besoin.
    rows = []
    for k in k_range:
        model = KMedoids(n_clusters=k, max_iter=100, random_state=random_state)
        labels = model.fit_predict(X)
        res = evaluate_clustering(X, labels, model_name=f"K-Medoids (k={k})")
        res["k"] = k
        rows.append(res)
    df = pd.DataFrame(rows)
    df["sil_norm"] = _min_max_norm(df["silhouette"])
    df["db_norm"] = _min_max_norm(df["davies_bouldin"], invert=True)
    df["ch_norm"] = _min_max_norm(df["calinski_harabasz"])
    df["score_global"] = (df["sil_norm"] + df["db_norm"] + df["ch_norm"]) / 3
    return df


def run_hac_ward(X, k: int, sample_size: int = 500, random_state=42):
    """HAC-Ward -- k est fixé a priori (suggéré par le dendrogramme dans le
    TFC original) ; ici on utilise le k retenu par K-Means pour cohérence,
    ou on peut fournir un k différent."""
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    res = evaluate_clustering(X, labels, model_name=f"HAC-Ward (k={k})")
    res["k"] = k
    return res, labels


def run_cycle1(X, k_range=range(2, 11), random_state=42) -> dict:
    """Exécute K-Means, K-Medoids et HAC-Ward, retourne les tableaux de
    résultats + le meilleur k par algorithme (identique en substance à la
    section 4.4.2 / 5.1 du TFC, généralisé pour être appelable en boucle)."""
    kmeans_df = run_kmeans_sweep(X, k_range=k_range, random_state=random_state)
    kmedoids_df = run_kmedoids_sweep(X, k_range=range(2, min(7, max(k_range))), random_state=random_state)

    best_k_kmeans = int(kmeans_df.loc[kmeans_df["score_global"].idxmax(), "k"])
    hac_res, hac_labels = run_hac_ward(X, k=best_k_kmeans, random_state=random_state)

    best_kmeans_row = kmeans_df.loc[kmeans_df["score_global"].idxmax()].to_dict()
    best_kmedoids_row = kmedoids_df.loc[kmedoids_df["score_global"].idxmax()].to_dict()

    return {
        "kmeans_df": kmeans_df,
        "kmedoids_df": kmedoids_df,
        "hac_result": hac_res,
        "best_kmeans": best_kmeans_row,
        "best_kmedoids": best_kmedoids_row,
        "best_hac": hac_res,
    }


# ---------------------------------------------------------------------------
# CYCLE 2 -- GMM, DBSCAN (identique au TFC, section 4.4.3 / 5.2)
# ---------------------------------------------------------------------------

def run_gmm_sweep(X, k_range=range(2, 11), random_state=42) -> pd.DataFrame:
    covariance_types = ["full", "tied", "diag", "spherical"]
    rows = []
    for k in k_range:
        for cov_type in covariance_types:
            try:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=random_state,
                    max_iter=200,
                    n_init=5,
                )
                model.fit(X)
                labels = model.predict(X)
                res = evaluate_clustering(X, labels, model_name=f"GMM (k={k}, cov={cov_type})")
                res["k"] = k
                res["covariance_type"] = cov_type
                res["bic"] = model.bic(X)
                res["aic"] = model.aic(X)
                res["log_likelihood"] = model.lower_bound_
                rows.append(res)
            except Exception:
                continue

    df = pd.DataFrame(rows)
    df["bic_norm"] = _min_max_norm(df["bic"], invert=True)
    df["sil_norm"] = _min_max_norm(df["silhouette"])
    df["score_global"] = (df["bic_norm"] + df["sil_norm"]) / 2
    return df


def run_dbscan_grid(
    X,
    eps_grid=None,
    min_samples_grid=(3, 5, 7, 10),
    max_noise_pct: float = 50.0,
) -> pd.DataFrame:
    if eps_grid is None:
        eps_grid = np.round(np.linspace(0.3, 2.0, 10), 2)

    rows = []
    n = X.shape[0]
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = 100.0 * np.sum(labels == -1) / n
            if n_clusters < 2 or noise_pct > max_noise_pct:
                continue
            res = evaluate_clustering(X, labels, model_name=f"DBSCAN (eps={eps}, min_samples={min_samples})")
            res["eps"] = eps
            res["min_samples"] = min_samples
            res["noise_pct"] = noise_pct
            rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["sil_norm"] = _min_max_norm(df["silhouette"])
    df["db_norm"] = _min_max_norm(df["davies_bouldin"], invert=True)
    df["ch_norm"] = _min_max_norm(df["calinski_harabasz"])
    df["score_global"] = (df["sil_norm"] + df["db_norm"] + df["ch_norm"]) / 3
    return df


def run_cycle2(X, k_range=range(2, 11), random_state=42) -> dict:
    gmm_df = run_gmm_sweep(X, k_range=k_range, random_state=random_state)
    dbscan_df = run_dbscan_grid(X)

    best_gmm_row = gmm_df.loc[gmm_df["score_global"].idxmax()].to_dict()
    best_dbscan_row = (
        dbscan_df.loc[dbscan_df["score_global"].idxmax()].to_dict()
        if not dbscan_df.empty
        else None
    )

    return {
        "gmm_df": gmm_df,
        "dbscan_df": dbscan_df,
        "best_gmm": best_gmm_row,
        "best_dbscan": best_dbscan_row,
    }


# ---------------------------------------------------------------------------
# Sélection finale entre familles d'algorithmes
# ---------------------------------------------------------------------------

def select_best_across_algorithms(cycle1_results: dict, cycle2_results: dict) -> dict:
    """Compare les meilleurs modèles de chaque famille sur le Silhouette
    Score BRUT (non renormalisé) -- voir note de rigueur en tête de module."""
    candidates = [
        ("K-Means", cycle1_results["best_kmeans"]),
        ("K-Medoids", cycle1_results["best_kmedoids"]),
        ("HAC-Ward", cycle1_results["best_hac"]),
        ("GMM", cycle2_results["best_gmm"]),
    ]
    if cycle2_results.get("best_dbscan") is not None:
        candidates.append(("DBSCAN", cycle2_results["best_dbscan"]))

    candidates = [(name, res) for name, res in candidates if not np.isnan(res.get("silhouette", np.nan))]
    if not candidates:
        raise ValueError("Aucun modèle valide à comparer.")

    best_name, best_res = max(candidates, key=lambda c: c[1]["silhouette"])
    return {"best_algorithm": best_name, **best_res}


def refit_final_model(X, algorithm: str, params: dict, random_state=42):
    """Ré-entraîne le modèle final retenu avec des initialisations plus
    nombreuses (n_init=10), comme dans le TFC original."""
    if algorithm == "K-Means":
        model = KMeans(n_clusters=int(params["k"]), random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
    elif algorithm == "K-Medoids":
        model = KMedoids(n_clusters=int(params["k"]), max_iter=300, random_state=random_state)
        labels = model.fit_predict(X)
    elif algorithm == "HAC-Ward":
        model = AgglomerativeClustering(n_clusters=int(params["k"]), linkage="ward")
        labels = model.fit_predict(X)
    elif algorithm == "GMM":
        model = GaussianMixture(
            n_components=int(params["k"]),
            covariance_type=params["covariance_type"],
            random_state=random_state,
            max_iter=200,
            n_init=10,
        )
        model.fit(X)
        labels = model.predict(X)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=int(params["min_samples"]))
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Algorithme inconnu : {algorithm}")

    return model, labels


def reorder_labels_by_size(labels: np.ndarray) -> np.ndarray:
    """Réordonne les labels de cluster par taille croissante (0 = plus
    petit), comme systématiquement fait dans le TFC pour la cohérence
    visuelle entre figures."""
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    sorted_idx = unique[np.argsort(counts)]
    mapping = {old: new for new, old in enumerate(sorted_idx)}
    mapping[-1] = -1
    return np.array([mapping.get(l, l) for l in labels])
