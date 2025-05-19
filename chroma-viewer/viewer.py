import chromadb
from chromadb.config import Settings
import pandas as pd
import streamlit as st
import numpy as np

# Configuración de pandas
pd.set_option('display.max_columns', 6)

# Carga la colección y guarda en session_state
def view_collections(db_path: str):
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("documents")
        collection_data = collection.get(include=["embeddings", "metadatas", "documents"])

        ids = collection_data.get("ids") or []
        documents = collection_data.get("documents") or []
        metadata = collection_data.get("metadatas") or []

        embeddings = collection_data.get("embeddings")
        if embeddings is None:
            embeddings = []
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        min_len = min(len(ids), len(embeddings), len(metadata), len(documents))
        data = {
            "ids": ids[:min_len],
            "documents": documents[:min_len],
            "metadata": metadata[:min_len],
            "embeddings": embeddings[:min_len],
        }

        st.session_state["chroma_data"] = data
        st.success(f"✅ Se cargaron {min_len} documentos.")

    except Exception as e:
        st.error(f"❌ Error al cargar la colección: {e}")

# Título principal
st.title("📂 Visualizador de ChromaDB")

# Entrada de ruta
db_path = st.text_input("📁 Ruta de la base de datos ChromaDB", "./chroma_db")

# Botón para cargar
if st.button("🔄 Cargar Colección"):
    if db_path.strip():
        view_collections(db_path.strip())
    else:
        st.error("⚠️ Por favor, ingrese una ruta válida para la base de datos.")

# Mostrar datos si están en session_state
if "chroma_data" in st.session_state:
    data = st.session_state["chroma_data"]

    df = pd.DataFrame({
        "IDs": data["ids"],
        "Metadata": data["metadata"],
        "Documents": data["documents"],
        "Embedding Length": [len(e) if e else 0 for e in data["embeddings"]],
        "Embedding (preview)": [e[:5] if e else [] for e in data["embeddings"]],
    })
    st.subheader("📋 Datos cargados")
    st.dataframe(df)

    selected_id = st.selectbox("🔎 Selecciona un ID para ver el embedding completo", data["ids"])
    if selected_id:
        idx = data["ids"].index(selected_id)
        st.subheader("📌 Embedding completo")
        st.json(data["embeddings"][idx])
