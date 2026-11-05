import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import (
    cluster_metrics,
    k_distances,
    prepare_features,
    project_2d,
    run_dbscan,
    run_hierarchical,
    sample_dataset,
    suggest_eps,
)

st.set_page_config(page_title="Clustering App", page_icon="🔵", layout="wide")
st.title("Clustering App")
st.caption("DBSCAN y Clustering Jerárquico sobre tu propio CSV.")

# ---------- 1. Data input ----------
st.header("1. Carga de datos")
file = st.file_uploader("Sube un archivo CSV", type=["csv"])
use_sample = st.checkbox("Usar dataset de ejemplo (si no tienes un CSV a la mano)")

if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:  # archivo mal formado
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()
elif use_sample:
    kind = st.selectbox("Dataset de ejemplo", ["Lunas", "Grupos gaussianos"])
    df = sample_dataset(kind)
else:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

if df.empty:
    st.error("El archivo no tiene filas.")
    st.stop()

st.subheader("Vista previa del dataset")
st.dataframe(df.head())
st.caption(f"{df.shape[0]} filas × {df.shape[1]} columnas")

# ---------- 2. Feature selection ----------
st.header("2. Selección de variables")
st.radio("Modo de selección", ["Manual (mínimo 2 variables)"], horizontal=True)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
features = st.multiselect(
    "Variables para el clustering",
    df.columns.tolist(),
    default=numeric_cols[:2] if len(numeric_cols) >= 2 else [],
)
if len(features) < 2:
    st.warning("Selecciona al menos 2 variables.")
    st.stop()

# ---------- 3. Preprocessing ----------
st.header("3. Preprocesamiento")
scale = st.checkbox("Escalar variables numéricas (StandardScaler)", value=True)
X, kept_index, feat_names = prepare_features(df, features, scale=scale)
dropped = len(df) - len(kept_index)
cat_selected = [c for c in features if c not in numeric_cols]
st.write(
    f"- Variables numéricas: **{[c for c in features if c in numeric_cols] or 'ninguna'}**"
    f"{' (escaladas)' if scale else ''}\n"
    f"- Variables categóricas codificadas con one-hot: **{cat_selected or 'ninguna'}**"
)
if dropped:
    st.info(f"Se descartaron {dropped} filas con valores faltantes en las variables elegidas.")
if len(X) < 5:
    st.error("Quedan muy pocas filas válidas para hacer clustering.")
    st.stop()
with st.expander("Ver datos preprocesados"):
    st.dataframe(pd.DataFrame(X, columns=feat_names, index=kept_index).head())

# ---------- 4. Model selection + hyperparameters ----------
st.header("4. Modelo e hiperparámetros")
model_name = st.selectbox("Algoritmo de clustering", ["DBSCAN", "Hierarchical Clustering"])

kdist = None
if model_name == "DBSCAN":
    c1, c2, c3 = st.columns(3)
    metric = c3.selectbox("Distancia", ["euclidean", "manhattan", "chebyshev"])
    min_samples = c2.slider("min_samples", 2, 30, 5)
    kdist = k_distances(X, min_samples, metric)
    eps_hint = suggest_eps(kdist)
    eps_max = round(float(max(kdist.max(), 0.5)), 2)
    eps = c1.slider(
        "eps",
        min_value=0.05,
        max_value=eps_max,
        value=float(np.clip(round(eps_hint, 2), 0.05, eps_max)),
        step=0.01,
        help=f"Sugerido por el gráfico k-distancia: {eps_hint:.2f}",
    )
    labels = run_dbscan(X, eps, min_samples, metric)
else:
    c1, c2 = st.columns(2)
    n_clusters = c1.slider("n_clusters", 2, min(10, len(X) - 1), 3)
    linkage = c2.selectbox("linkage", ["ward", "complete", "average", "single"])
    labels = run_hierarchical(X, n_clusters, linkage)

# ---------- 5. Output ----------
st.header("5. Resultados")
result = df.loc[kept_index].copy()
result["cluster"] = labels

st.subheader("Etiquetas de cluster")
counts = (
    pd.Series(labels, name="cluster")
    .value_counts()
    .sort_index()
    .rename_axis("cluster")
    .reset_index(name="puntos")
)
if model_name == "DBSCAN":
    counts["cluster"] = counts["cluster"].astype(str).replace({"-1": "-1 (ruido)"})
st.dataframe(counts, hide_index=True)

st.subheader("Dataset con clusters")
st.dataframe(result)
st.download_button(
    "Descargar dataset con clusters (CSV)",
    result.to_csv(index=False).encode("utf-8"),
    file_name="dataset_clusters.csv",
    mime="text/csv",
)

# ---------- 6. Visualization ----------
st.header("6. Visualización")
noise = labels == -1
both_numeric = len(features) == 2 and all(f in numeric_cols for f in features)

fig, ax = plt.subplots(figsize=(7, 5))
if both_numeric:
    Z = result[features].to_numpy(dtype=float)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
else:
    Z, how = project_2d(X)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    st.info(
        "El scatter plot 2D con los valores originales se muestra al seleccionar exactamente "
        f"2 variables numéricas. Aquí se usa una proyección PCA: {how}."
    )
if noise.any():
    ax.scatter(Z[noise, 0], Z[noise, 1], c="lightgray", marker="x", s=25, label="Ruido")
for lab in sorted(set(labels) - {-1}):
    pts = labels == lab
    ax.scatter(Z[pts, 0], Z[pts, 1], s=25, label=f"Cluster {lab}")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

if kdist is not None:
    with st.expander("Gráfico k-distancia (ayuda para elegir eps)"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(kdist)
        ax.axhline(eps, color="tab:red", ls="--", label=f"eps = {eps:.2f}")
        ax.set_xlabel("Puntos ordenados por distancia")
        ax.set_ylabel(f"Distancia al vecino #{min_samples}")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Un buen eps suele estar en el 'codo' de la curva.")

# ---------- 7. Evaluation ----------
st.header("7. Evaluación")
m = cluster_metrics(X, labels)
e1, e2, e3, e4 = st.columns(4)
e1.metric("Clusters", m["n_clusters"])
e2.metric("Puntos de ruido", m["n_noise"])
e3.metric("Silhouette Score", "—" if m["silhouette"] is None else f"{m['silhouette']:.3f}")
e4.metric(
    "Davies-Bouldin Index", "—" if m["davies_bouldin"] is None else f"{m['davies_bouldin']:.3f}"
)
st.caption(
    "Silhouette: más alto es mejor (rango -1 a 1). Davies-Bouldin: más bajo es mejor. "
    + ("En DBSCAN las métricas se calculan sin los puntos de ruido." if model_name == "DBSCAN" else "")
)
if m["n_clusters"] < 2:
    st.warning(
        "Se necesitan al menos 2 clusters para calcular las métricas. "
        "En DBSCAN prueba subir eps o bajar min_samples."
    )
