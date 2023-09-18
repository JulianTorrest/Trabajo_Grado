# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Función para obtener datos desde Google Sheets
def get_data_from_gsheets(sheet_url):
    # Credenciales y autenticación
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_credentials.json', scope)
    client = gspread.authorize(creds)
    
    # Obtener el documento y la primera hoja
    sheet = client.open_by_url(sheet_url).sheet1

    # Convertir los datos a DataFrame
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Iniciar la app Streamlit
st.title('Análisis Exploratorio de Datos y Clustering desde Google Sheets')

# Cargar los datos
url = "https://docs.google.com/spreadsheets/d/1r4YcJuh5Qvp9_Z9D4soEyZymZD6tGTYBqqevXTIT6AQ/edit#gid=0"
try:
    data = get_data_from_gsheets(url)
    st.write("Datos cargados exitosamente!")
    st.write(data.head())
except Exception as e:
    st.write("Hubo un error al cargar los datos.")
    st.write(e)

# Si los datos se cargaron, realizar el clustering y visualización
if 'data' in locals():
    metrics = ['ROE', 'ROA', 'EBITDA', 'APALANCAMIENTO', 'ACTIVOS', 'PASIVOS', 'PATRIMONIO', 
               'INGRESOS DE ACTIVIDADES ORDINARIAS', 'GANANCIA BRUTA', 'GANANCIA (PÉRDIDA) POR ACTIVIDADES DE OPERACIÓN', 'GANANCIA (PÉRDIDA)']
    
    st.subheader('Clustering basado en métricas financieras')
    num_clusters = st.slider("Selecciona el número de clusters", 2, 10, 3)
    
    # 1. Preprocesamiento
    df_metrics = data[metrics].dropna()  # Eliminar cualquier fila con datos faltantes
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_metrics)
    
    # 2. Modelo KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_metrics['cluster'] = kmeans.fit_predict(scaled_data)
    
    # 3. Visualización
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

# Para ejecutar el código:
# 1. Guarda este código en un archivo, por ejemplo "app.py".
# 2. Instala las dependencias con pip.
# 3. Ejecuta el comando: streamlit run app.py


