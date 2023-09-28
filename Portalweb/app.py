import streamlit as st
import requests
from PIL import Image
from io import BytesIO

def get_image_from_github(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def main():
    st.set_page_config(page_title="Hub de Bioeconomía", page_icon=":leaf:")

    st.title("Hub de Bioeconomía")
    st.subheader("Datos Generales")

    base_url = "https://raw.githubusercontent.com/"

    # Lista de rutas de las imágenes en GitHub
    paths = [
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Datos generales Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Inflación.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Desempleo.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/PIB Colombia 2023.png",
    ]
    
    for path in paths:
        try:
            image_url = base_url + path
            image = get_image_from_github(image_url)
            st.write(path.split("/")[-1].replace(".png", ""))  # Título de la imagen basado en el nombre del archivo
            st.image(image, use_column_width=True)
        except Exception as e:
            st.write(f"Ha ocurrido un error al cargar la imagen {path}: {e}")

if __name__ == "__main__":
    main()

