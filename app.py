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
    sample_dataset,
    suggest_eps,
)

st.set_page_config(page_title="Clustering con DBSCAN", page_icon="🔵", layout="wide")
st.title("Clustering con DBSCAN")
st.caption("Agrupamiento por densidad: encuentra clusters de forma libre y marca el ruido.")

# ---------- Datos ----------
with st.sidebar:
    st.header("1. Datos")
    source = st.radio("Origen", ["Dataset de ejemplo", "Subir CSV"])
    if source == "Subir CSV":
        file = st.file_uploader("Archivo CSV", type=["csv"])
        if file is None:
            st.info("Sube un archivo CSV para comenzar.")
            st.stop()
        try:
            df = pd.read_csv(file)
        except Exception as e:  # archivo mal formado
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()
    else:
        kind = st.selectbox("Ejemplo", ["Lunas", "Grupos gaussianos"])
        df = sample_dataset(kind)

if df.empty:
    st.error("El archivo no tiene filas.")
    st.stop()

with st.sidebar:
    cols = df.columns.tolist()
    numeric = df.select_dtypes(include="number").columns.tolist()
    default = numeric[:2] if len(numeric) >= 2 else cols[:2]
    features = st.multiselect("Variables a usar", cols, default=default)
    scale = st.checkbox("Estandarizar variables numéricas", value=True)

if len(features) < 2:
    st.warning("Selecciona al menos 2 variables en el panel lateral.")
    st.stop()

X, kept_index, feat_names = prepare_features(df, features, scale=scale)
dropped = len(df) - len(kept_index)
if len(X) < 5:
    st.error("Quedan muy pocas filas válidas después de quitar valores faltantes.")
    st.stop()

# ---------- Parámetros ----------
with st.sidebar:
    st.header("2. Parámetros")
    metric = st.selectbox("Distancia", ["euclidean", "manhattan", "chebyshev"])
    min_samples = st.slider("min_samples", 2, 30, 5)

kdist = k_distances(X, min_samples, metric)
eps_hint = suggest_eps(kdist)
eps_max = float(max(kdist.max(), 0.5))

with st.sidebar:
    eps = st.slider(
        "eps (radio del vecindario)",
        min_value=0.05,
        max_value=round(eps_max, 2),
        value=float(np.clip(round(eps_hint, 2), 0.05, round(eps_max, 2))),
        step=0.01,
    )
    st.caption(f"eps sugerido por el gráfico k-distancia: {eps_hint:.2f}")

labels = run_dbscan(X, eps, min_samples, metric)
m = cluster_metrics(X, labels)

# ---------- Resultados ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clusters", m["n_clusters"])
c2.metric("Puntos de ruido", m["n_noise"], f"{m['noise_pct']:.1f}%", delta_color="off")
c3.metric("Silhouette", "—" if m["silhouette"] is None else f"{m['silhouette']:.3f}")
c4.metric("Davies-Bouldin", "—" if m["davies_bouldin"] is None else f"{m['davies_bouldin']:.3f}")

if m["n_clusters"] < 2:
    st.warning(
        "DBSCAN encontró menos de 2 clusters, así que no se pueden calcular las métricas. "
        "Prueba subir eps o bajar min_samples."
    )
if dropped:
    st.info(f"Se descartaron {dropped} filas con valores faltantes.")

tab_plot, tab_k, tab_data = st.tabs(["Clusters", "Gráfico k-distancia", "Datos"])

with tab_plot:
    Z, how = project_2d(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    noise = labels == -1
    ax.scatter(Z[noise, 0], Z[noise, 1], c="lightgray", marker="x", s=25, label="Ruido")
    for lab in sorted(set(labels) - {-1}):
        pts = labels == lab
        ax.scatter(Z[pts, 0], Z[pts, 1], s=25, label=f"Cluster {lab}")
    if how == "directo":
        ax.set_xlabel(feat_names[0])
        ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else "")
    else:
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        st.caption(f"Proyección: {how}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with tab_k:
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

with tab_data:
    result = df.loc[kept_index].copy()
    result["cluster"] = labels
    st.dataframe(result)
    st.download_button(
        "Descargar resultado (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="resultado_dbscan.csv",
        mime="text/csv",
    )
