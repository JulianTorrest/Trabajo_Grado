# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Función para obtener datos desde Google Sheets
def get_data_from_gsheets(sheet_url):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Función para descargar datos en CSV
def download_link_csv(object_to_download, download_filename, download_link_text):
    csv = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Streamlit
st.title('Análisis Exploratorio de Datos y Clustering desde Google Sheets')
url = "https://docs.google.com/spreadsheets/d/1r4YcJuh5Qvp9_Z9D4soEyZymZD6tGTYBqqevXTIT6AQ/edit#gid=0"

try:
    data = get_data_from_gsheets(url)
    st.write("Datos cargados exitosamente!")
    st.write(data.head())
except Exception as e:
    st.write("Hubo un error al cargar los datos.")
    st.write(e)

if 'data' in locals():
    st.subheader('Seleccionar Información')
    razon_social = st.multiselect('RAZÓN SOCIAL', data['RAZÓN SOCIAL'].unique())
    subsector_options = [
        "", 
        "B0910 - ACTIVIDADES DE APOYO PARA LA EXTRACCIÓN DE PETRÓLEO Y DE GAS NATURAL",
        "B0812 - EXTRACCIÓN DE ARCILLAS DE USO INDUSTRIAL, CALIZA, CAOLÍN Y BENTONITAS",
        "B0610 - EXTRACCIÓN DE PETRÓLEO CRUDO",
        "B0620 - EXTRACCIÓN DE GAS NATURAL",
        "B0510 - EXTRACCIÓN DE HULLA (CARBÓN DE PIEDRA)"
        "B0722 - EXTRACCIÓN DE ORO Y OTROS METALES PRECIOSOS",
        "B0811 - EXTRACCIÓN DE PIEDRA, ARENA, ARCILLAS COMUNES, YESO Y ANHIDRITA",
        "B0820 - EXTRACCIÓN DE ESMERALDAS, PIEDRAS PRECIOSAS Y SEMIPRECIOSAS",
        "B0729 - EXTRACCIÓN DE OTROS MINERALES METALÍFEROS NO FERROSOS N.C.P.",
        "B0990 - ACTIVIDADES DE APOYO PARA OTRAS ACTIVIDADES DE EXPLOTACIÓN DE MINAS Y CANTERAS",
        "B0723 - EXTRACCIÓN DE MINERALES DE NÍQUEL",
        "B0899 - EXTRACCIÓN DE OTROS MINERALES NO METÁLICOS N.C.P.",
        "B0892 - EXTRACCIÓN DE HALITA (SAL)",
        "B0891 - EXTRACCIÓN DE MINERALES PARA LA FABRICACIÓN DE ABONOS Y PRODUCTOS QUÍMICOS",
        "B0710 - EXTRACCIÓN DE MINERALES DE HIERRO",
        "C3250 - FABRICACIÓN DE INSTRUMENTOS, APARATOS Y MATERIALES MÉDICOS Y ODONTOLÓGICOS (INCLUIDO MOBILIARIO)",
        "C1030 - ELABORACIÓN DE ACEITES Y GRASAS DE ORIGEN VEGETAL Y ANIMAL",
        "C1410 - CONFECCIÓN DE PRENDAS DE VESTIR, EXCEPTO PRENDAS DE PIEL",
        "C1051 - ELABORACIÓN DE PRODUCTOS DE MOLINERÍA",
        "C2229 - FABRICACIÓN DE ARTÍCULOS DE PLÁSTICO N.C.P."
        "C1709 - FABRICACIÓN DE OTROS ARTÍCULOS DE PAPEL Y CARTÓN",
        "C2930 - FABRICACIÓN DE PARTES, PIEZAS (AUTOPARTES) Y ACCESORIOS (LUJOS) PARA VEHÍCULOS AUTOMOTORES",
        "C2011 - FABRICACIÓN DE SUSTANCIAS Y PRODUCTOS QUÍMICOS BÁSICOS",
        "C1011 - PROCESAMIENTO Y CONSERVACIÓN DE CARNE Y PRODUCTOS CÁRNICOS",
        "C1089 - ELABORACIÓN DE OTROS PRODUCTOS ALIMENTICIOS N.C.P.",
        "C1811 - ACTIVIDADES DE IMPRESIÓN",
        "C2599 - FABRICACIÓN DE OTROS PRODUCTOS ELABORADOS DE METAL N.C.P.",
        "C2100 - FABRICACIÓN DE PRODUCTOS FARMACÉUTICOS, SUSTANCIAS QUÍMICAS MEDICINALES Y PRODUCTOS BOTÁNICOS DE USO FARMACÉUTICO",
        "C3091 - FABRICACIÓN DE MOTOCICLETAS",
        "C2395 - FABRICACIÓN DE ARTÍCULOS DE HORMIGÓN, CEMENTO Y YESO",
        "C2221 - FABRICACIÓN DE FORMAS BÁSICAS DE PLÁSTICO",
        "C1812 - ACTIVIDADES DE SERVICIOS RELACIONADOS CON LA IMPRESIÓN",
        "C2029 - FABRICACIÓN DE OTROS PRODUCTOS QUÍMICOS N.C.P.",
        "C1311 - PREPARACIÓN E HILATURA DE FIBRAS TEXTILES",
        "C1522 - FABRICACIÓN DE OTROS TIPOS DE CALZADO, EXCEPTO CALZADO DE CUERO Y PIEL",
        "C2750 - FABRICACIÓN DE APARATOS DE USO DOMÉSTICO",
        "C1392 - CONFECCIÓN DE ARTÍCULOS CON MATERIALES TEXTILES, EXCEPTO PRENDAS DE VESTIR",
        "C3011 - CONSTRUCCIÓN DE BARCOS Y DE ESTRUCTURAS FLOTANTES",
        "C1090 - ELABORACIÓN DE ALIMENTOS PREPARADOS PARA ANIMALES",
        "C2593 - FABRICACIÓN DE ARTÍCULOS DE CUCHILLERÍA, HERRAMIENTAS DE MANO Y ARTÍCULOS DE FERRETERÍA"
        "C2021 - FABRICACIÓN DE PLAGUICIDAS Y OTROS PRODUCTOS QUÍMICOS DE USO AGROPECUARIO",
        "C2731 - FABRICACIÓN DE HILOS Y CABLES ELÉCTRICOS Y DE FIBRA ÓPTICA",
        "C1061 - TRILLA DE CAFÉ",
        "C1040 - ELABORACIÓN DE PRODUCTOS LÁCTEOS",
        "C1020 - PROCESAMIENTO Y CONSERVACIÓN DE FRUTAS, LEGUMBRES, HORTALIZAS Y TUBÉRCULOS",
        "C2022 - FABRICACIÓN DE PINTURAS, BARNICES Y REVESTIMIENTOS SIMILARES, TINTAS PARA IMPRESIÓN Y MASILLAS",
        "C3210 - FABRICACIÓN DE JOYAS, BISUTERÍA Y ARTÍCULOS CONEXOS",
        "C2829 - FABRICACIÓN DE OTROS TIPOS DE MAQUINARIA Y EQUIPO DE USO ESPECIAL N.C.P.",
        "C1620 - FABRICACIÓN DE HOJAS DE MADERA PARA ENCHAPADO; FABRICACIÓN DE TABLEROS CONTRACHAPADOS, TABLEROS LAMINADOS, TABLEROS DE PARTÍCULAS Y OTROS TABLEROS Y PANELES",
        "C1012 - PROCESAMIENTO Y CONSERVACIÓN DE PESCADOS, CRUSTÁCEOS Y MOLUSCOS",
        "C2410 - INDUSTRIAS BÁSICAS DE HIERRO Y DE ACERO",
        "C2821 - FABRICACIÓN DE MAQUINARIA AGROPECUARIA Y FORESTAL",
        "C1104 - ELABORACIÓN DE BEBIDAS NO ALCOHÓLICAS, PRODUCCIÓN DE AGUAS MINERALES Y DE OTRAS AGUAS EMBOTELLADAS",
        "C1910 - FABRICACIÓN DE PRODUCTOS DE HORNOS DE COQUE",
        "C1702 - FABRICACIÓN DE PAPEL Y CARTÓN ONDULADO (CORRUGADO); FABRICACIÓN DE ENVASES, EMPAQUES Y DE EMBALAJES DE PAPEL Y CARTÓN.",
        "C1513 - FABRICACIÓN DE ARTÍCULOS DE VIAJE, BOLSOS DE MANO Y ARTÍCULOS SIMILARES; ARTÍCULOS DE TALABARTERÍA Y GUARNICIONERÍA ELABORADOS EN OTROS MATERIALES",
        "C3290 - OTRAS INDUSTRIAS MANUFACTURERAS N.C.P.",
        "C2012 - FABRICACIÓN DE ABONOS Y COMPUESTOS INORGÁNICOS NITROGENADOS",
        "C2023 - FABRICACIÓN DE JABONES Y DETERGENTES, PREPARADOS PARA LIMPIAR Y PULIR; PERFUMES Y PREPARADOS DE TOCADOR",
        "C2511 - FABRICACIÓN DE PRODUCTOS METÁLICOS PARA USO ESTRUCTURAL",
        "C2711 - FABRICACIÓN DE MOTORES, GENERADORES Y TRANSFORMADORES ELÉCTRICOS",
        "C3110 - FABRICACIÓN DE MUEBLES",
        "C1921 - FABRICACIÓN DE PRODUCTOS DE LA REFINACIÓN DEL PETRÓLEO",
        "C1701 - FABRICACIÓN DE PULPAS (PASTAS) CELULÓSICAS; PAPEL Y CARTÓN",
        "C1313 - ACABADO DE PRODUCTOS TEXTILES"]
    subsector = st.selectbox('SUBSECTOR', subsector_options)

    sector_options = ["", 
                      "EXPLOTACIÓN DE MINAS Y CANTERAS", 
                      "INDUSTRIAS MANUFACTURERAS", 
                      "CONSTRUCCIÓN", 
                      "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACIÓN DE VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS", 
                      "AGRICULTURA, GANADERÍA, CAZA, SILVICULTURA Y PESCA"]
    sector = st.selectbox('SECTOR', sector_options)

    macrosector = st.selectbox('MACROSECTOR', ["", "MINERÍA", "MANUFACTURERO", "CONSTRUCCIÓN", "COMERCIO", "AGROPECUARIO"])

    if st.button('Ejecutar'):
        if razon_social:
            data = data[data['RAZÓN SOCIAL'].isin(razon_social)]
        if subsector:
            data = data[data['SUBSECTOR'] == subsector] # Corregido el filtro
        if sector:
            data = data[data['SECTOR'] == sector] # Corregido el filtro
        if macrosector:
            data = data[data['MACROSECTOR'] == macrosector]

        st.write(data.head())


        # Clustering
        metrics = ['ROE', 'ROA', 'EBITDA', 'APALANCAMIENTO', 'ACTIVOS', 'PASIVOS', 'PATRIMONIO', 
                   'INGRESOS DE ACTIVIDADES ORDINARIAS', 'GANANCIA BRUTA', 'GANANCIA (PÉRDIDA) POR ACTIVIDADES DE OPERACIÓN', 'GANANCIA (PÉRDIDA)']
        
        num_clusters = st.slider("Selecciona el número de clusters", 2, 10, 3)
        df_metrics = data[metrics].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_metrics)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df_metrics['cluster'] = kmeans.fit_predict(scaled_data)

        # Métricas de Evaluación
        # Métricas de Evaluación
        st.subheader('Métricas de Evaluación del Modelo')
        silhouette = silhouette_score(scaled_data, df_metrics['cluster'])
        inertia = kmeans.inertia_
        st.write(f'Coeficiente de silueta: {silhouette:.2f}')
        st.write(f'Inercia: {inertia:.2f}')

        pca = PCA(2)
        pca_results = pca.fit_transform(scaled_data)
        df_metrics['pca-1'] = pca_results[:, 0]
        df_metrics['pca-2'] = pca_results[:, 1]

        fig, ax = plt.subplots()
        sns.scatterplot(x='pca-1', y='pca-2', hue='cluster', data=df_metrics, ax=ax, palette="deep", s=100)
        st.pyplot(fig)

        # Mapa de Calor
        st.subheader("Mapa de Calor para las Métricas")
        correlation = df_metrics[metrics].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Descargar datos
        if st.button('Descargar CSV'):
            tmp_download_link = download_link_csv(data, 'data.csv', 'Haz clic aquí para descargar en CSV')
            st.markdown(tmp_download_link, unsafe_allow_html=True)


# Para ejecutar el código:
# 1. Guarda este código en un archivo, por ejemplo "app.py".
# 2. Instala las dependencias con pip.
# 3. Ejecuta el comando: streamlit run app.py


