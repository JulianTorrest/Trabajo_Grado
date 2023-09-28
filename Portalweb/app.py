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
    default_path = "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Colombia.png"
    
    try:
        image_url = base_url + default_path
        image = get_image_from_github(image_url)
        st.write("Datos generales")  # Aquí agregas el título estático para la imagen
        st.image(image, use_column_width=True)
    except Exception as e:
        st.write(f"Ha ocurrido un error al cargar la imagen: {e}")

if __name__ == "__main__":
    main()
