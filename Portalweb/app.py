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
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Acuerdos comerciales de Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Inflación.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Desempleo.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Escalafón de competitividad Internacional.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/IED Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Inversión directa en el pais.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/IED por sectores.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/PIB Colombia 2023.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Composición del PIB 2021.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Revisiones de las actividades económicas.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Valor Agregado por actividad economica.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Crecimiento agropecuario.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/PIB Agropecuario total y por ramas.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Detalle de la producción de los principales productos de la actividad agricola.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Producción y variación en principales actividades agroindustriales.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Agricultura,ganaderia,caza,silvicultura,pesca.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Producción de carne de res,pollo y cerdo.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Precio promedio mensual de los productos de la cadena pecuaria.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Actividad pecuaria.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Producción y variación en principales actividades agroindustriales.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Actividades artisticias de entrenamiento.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Actividades profesionales,cientificas y tecnicas.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Comercio al por mayor y al por menor.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Contruccion de edificaciones.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Contrucción.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Electricidad,gas,vapor y aire acondicionado.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Explotación de minas y canteras.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Producción de explotación de minas y canteras.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Industrias Manufactureras.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Información y comunicaciones.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Consumo final por hogares.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Consumo final por durabilidad.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Comercio exterior Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Exportaciones.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Exportaciones de colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Principales productos de exportación.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Exportaciones de banano,flores y café.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Principales productos de importación de Colombia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Usaquen.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Chapinero.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Santa Fe.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia San Cristobal.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Usme.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Tunjuelito.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Bosa.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Kenedy.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Fontibon.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Engativa.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Suba.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Barrios Unidos.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Teusaquillo.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Los Martires.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Antonio Nariño.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia la Candelaria.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Ciudad Bolivar.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Sumapaz.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Laboral Bogota.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Laboral Bogota 2.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Cajica.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Chia.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Cota.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Facatativa.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Funza.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Fusagasuga.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia La Calera.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Madrid.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Mosquera.png",
        "JulianTorrest/Trabajo_Grado/main/Portalweb/Colombia/Infografia Soacha.png",
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

