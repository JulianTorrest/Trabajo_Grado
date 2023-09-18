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
    subsector = st.multiselect('SUBSECTOR', data['SUBSECTOR'].unique())
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
            data = data[data['SUBSECTOR'].isin(subsector)]
        if sector:
            data = data[data['SECTOR'].isin(sector)]
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
        st.subheader('Métricas de Evaluación del Modelo')
        silhouette = silhouette_score(scaled_data, df_metrics['cluster'])
        inertia = kmeans.inertia_
        st.write(f'Coeficiente de silueta: {silhouette:.2f}')
        st.write(f'Inercia: {inertia:.2f}')

        pca = PCA(2)
        pca_result = pca.fit_transform(scaled_data)
        df_metrics['pca_1'] = pca_result[:,0]
        df_metrics['pca_2'] = pca_result[:,1]
        
        chart = alt.Chart(df_metrics).mark_circle(size=60).encode(
            x='pca_1',
            y='pca_2',
            color='cluster:N',
            tooltip=['ROE', 'ROA', 'EBITDA']
        ).interactive()
        st.altair_chart(chart)

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


