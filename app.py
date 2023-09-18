# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt

# Definir función para obtener datos desde Google Sheets
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
st.title('Análisis Exploratorio de Datos desde Google Sheets')

# Cargar los datos
url = "https://docs.google.com/spreadsheets/d/1r4YcJuh5Qvp9_Z9D4soEyZymZD6tGTYBqqevXTIT6AQ/edit#gid=0"
try:
    data = get_data_from_gsheets(url)
    st.write("Datos cargados exitosamente!")
    st.write(data.head())
except Exception as e:
    st.write("Hubo un error al cargar los datos.")
    st.write(e)

# Si los datos se cargaron, visualizar un histograma
if 'data' in locals():
    metrics = ['ROE', 'ROA', 'EBITDA', 'APALANCAMIENTO', 'ACTIVOS', 'PASIVOS', 'PATRIMONIO', 
               'INGRESOS DE ACTIVIDADES ORDINARIAS', 'GANANCIA BRUTA', 'GANANCIA (PÉRDIDA) POR ACTIVIDADES DE OPERACIÓN', 'GANANCIA (PÉRDIDA)']
    
    column_to_plot = st.selectbox('Selecciona una métrica para visualizar', metrics)
    if pd.api.types.is_numeric_dtype(data[column_to_plot]):
        histogram = alt.Chart(data).mark_bar().encode(
            alt.X(column_to_plot, bin=True),
            y='count()',
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(histogram)
    else:
        st.write(f"La métrica {column_to_plot} no es numérica y no puede visualizarse como histograma.")

# Instrucciones para ejecutar:
# 1. Guarda este código en un archivo, por ejemplo "app.py".
# 2. En tu terminal o línea de comandos, navega al directorio donde guardaste el archivo.
# 3. Ejecuta el comando: streamlit run app.py

