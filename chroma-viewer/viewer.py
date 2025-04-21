import chromadb
from chromadb.config import Settings
import pandas as pd
import streamlit as st

# Configuración de pandas para mostrar más columnas
pd.set_option('display.max_columns', 4)

def view_collections(db_path):
    #st.markdown(f"### DB Path: {db_path}")

    try:
        # Crear cliente persistente de ChromaDB
        client = chromadb.PersistentClient(path=db_path)

        # Intentar cargar la colección "documents"
        #st.write("Cargando colección...")
        collection = client.get_collection("documents").get()
        #st.write("Colección cargada exitosamente.")

        # Mostrar datos de la colección
        ids = collection['ids']
        embeddings = collection["embeddings"]
        metadata = collection["metadatas"]
        documents = collection["documents"]

        # Crear un DataFrame para mostrar los datos
        df = pd.DataFrame({
            "IDs": ids,
            "Embeddings": embeddings,
            "Metadata": metadata,
            "Documents": documents
        })
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error al cargar la colección: {e}")

# Interfaz principal de Streamlit
st.title("ChromaDB Viewer")

# Entrada para la ruta de la base de datos
db_path = st.text_input("Ingrese la ruta de la base de datos ChromaDB", "./chroma_db")

if st.button("Cargar Colección"):
    if db_path:
        view_collections(db_path)
    else:
        st.error("Por favor, ingrese una ruta válida para la base de datos.")

