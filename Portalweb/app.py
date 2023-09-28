# Importando las bibliotecas necesarias
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Definición de función para obtener imágenes desde GitHub
def get_image_from_github(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Configuración de la página
st.set_page_config(page_title="Hub de Bioeconomía", page_icon=":leaf:")

# Título y subtítulo de la página
st.title("Hub de Bioeconomía")
st.subheader("Datos Generales")

# Entrada de la URL de la imagen
url_input = st.text_input("https://github.com/JulianTorrest/Trabajo_Grado/blob/main/Portalweb/Colombia/Colombia.png")

# Si se proporciona una URL, muestra la imagen
if url_input:
    try:
        image = get_image_from_github(url_input)
        st.image(image, caption="Imagen cargada desde GitHub", use_column_width=True)
    except Exception as e:
        st.write(f"Ha ocurrido un error al cargar la imagen: {e}")

# Puedes personalizar la aplicación con más características, como la posibilidad de elegir entre múltiples imágenes,
# procesar las imágenes, añadir más información, etc.

# Para correr tu aplicación, guarda el código en un archivo llamado app.py y en tu terminal ejecuta:
# streamlit run app.py

if __name__ == "__main__":
    main()
