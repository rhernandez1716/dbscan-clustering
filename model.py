"""Funciones de preparación de datos, DBSCAN y métricas para la app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def sample_dataset(name: str, n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Genera un dataset sintético para probar la app sin subir archivos."""
    rng = np.random.default_rng(seed)
    if name == "Lunas":
        X, _ = make_moons(n_samples=n, noise=0.07, random_state=seed)
    else:
        X, _ = make_blobs(
            n_samples=n,
            centers=[(-4, -2), (0, 4), (5, -1)],
            cluster_std=[0.8, 1.0, 0.7],
            random_state=seed,
        )
    # unos cuantos puntos dispersos para que haya ruido real
    outliers = rng.uniform(X.min(axis=0) - 1, X.max(axis=0) + 1, size=(n // 20, 2))
    X = np.vstack([X, outliers])
    return pd.DataFrame(X, columns=["x1", "x2"])


def prepare_features(
    df: pd.DataFrame, features: list[str], scale: bool = True
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """Construye la matriz numérica para el modelo.

    - Las columnas numéricas se estandarizan (opcional).
    - Las categóricas se convierten con one-hot.
    - Se descartan las filas con valores faltantes.

    Devuelve la matriz, el índice de las filas conservadas y los nombres
    de las columnas resultantes.
    """
    sub = df[features]
    num_cols = sub.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    parts = [sub[num_cols]]
    if cat_cols:
        parts.append(pd.get_dummies(sub[cat_cols].astype(str), dtype=float))
    full = pd.concat(parts, axis=1)

    valid = ~sub.isna().any(axis=1)
    full = full.loc[valid]

    if scale and num_cols:
        full[num_cols] = StandardScaler().fit_transform(full[num_cols])

    return full.to_numpy(dtype=float), full.index, full.columns.tolist()


def run_dbscan(X: np.ndarray, eps: float, min_samples: int, metric: str) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)


def k_distances(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Distancia al k-ésimo vecino de cada punto, ordenada de menor a mayor."""
    k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, _ = nn.kneighbors(X)
    return np.sort(dist[:, -1])


def suggest_eps(sorted_dist: np.ndarray) -> float:
    """Heurística del "codo": punto más alejado de la recta que une los extremos."""
    n = len(sorted_dist)
    if n < 3:
        return float(sorted_dist[-1])
    x = np.linspace(0, 1, n)
    y = (sorted_dist - sorted_dist[0]) / max(sorted_dist[-1] - sorted_dist[0], 1e-12)
    return float(sorted_dist[int(np.argmax(x - y))])


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Métricas de calidad calculadas solo sobre los puntos que no son ruido."""
    mask = labels != -1
    clusters = set(labels[mask])
    out = {
        "n_clusters": len(clusters),
        "n_noise": int((~mask).sum()),
        "noise_pct": float((~mask).mean() * 100),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(clusters) >= 2 and mask.sum() > len(clusters):
        out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    return out


def project_2d(X: np.ndarray) -> tuple[np.ndarray, str]:
    """Devuelve coordenadas 2D para graficar (PCA si hay más de 2 columnas)."""
    if X.shape[1] == 2:
        return X, "directo"
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros(len(X))]), "directo"
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum() * 100
    return Z, f"PCA ({var:.0f}% de la varianza)"
