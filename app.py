import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from fpdf import FPDF

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

def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Descargar CSV</a>'

# Función para descargar el DataFrame como PDF
def get_pdf_download_link(df):
    pdf = FPDF()
    pdf.add_page()
    page_width = pdf.w - 2 * pdf.l_margin
    
    col_width = page_width/len(df.columns)
    row_height = pdf.font_size
    
    for col in df.columns:
        pdf.cell(col_width, row_height*2, txt=col, border=1)
        
    pdf.ln(row_height*2)

    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height*2, txt=str(item), border=1)
            
    pdf.ln(row_height*2)
    filename = "data.pdf"
    pdf.output(filename).encode('latin1')
    
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<a href="data:file/pdf;base64,{b64}" download="data.pdf">Descargar PDF</a>'

# Streamlit
st.title('Análisis Exploratorio de Datos y Clustering desde Google Sheets')

url = "https://docs.google.com/spreadsheets/d/1r4YcJuh5Qvp9_Z9D4soEyZymZD6tGTYBqqevXTIT6AQ/edit#gid=0"
data = get_data_from_gsheets(url)

st.write("Datos cargados exitosamente!")
st.write(data.head())

razon_social = st.text_input('RAZÓN SOCIAL')
subsector = st.text_input('SUBSECTOR')
sector = st.text_input('SECTOR')
macrosector = st.text_input('MACROSECTOR')

metrics = ['ROE', 'ROA', 'EBITDA', 'APALANCAMIENTO', 'ACTIVOS', 'PASIVOS', 'PATRIMONIO', 
           'INGRESOS DE ACTIVIDADES ORDINARIAS', 'GANANCIA BRUTA', 'GANANCIA (PÉRDIDA) POR ACTIVIDADES DE OPERACIÓN', 'GANANCIA (PÉRDIDA)']

if st.button('Ejecutar'):
    # Filtro los datos
    filtered_data = data[
        (data['RAZÓN SOCIAL'].str.contains(razon_social)) & 
        (data['SUBSECTOR'].str.contains(subsector)) & 
        (data['SECTOR'].str.contains(sector)) & 
        (data['MACROSECTOR'].str.contains(macrosector))
    ]

    # Clustering
    df_metrics = filtered_data[metrics].dropna()  # Eliminar cualquier fila con datos faltantes
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_metrics)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_metrics['cluster'] = kmeans.fit_predict(scaled_data)

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
    st.subheader('Mapa de Calor')
    correlation_matrix = filtered_data[metrics].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Descargar en PDF y CSV
    st.markdown(get_csv_download_link(filtered_data), unsafe_allow_html=True)
    st.markdown(get_pdf_download_link(filtered_data), unsafe_allow_html=True)

# Para ejecutar el código:
# 1. Guarda este código en un archivo, por ejemplo "app.py".
# 2. Instala las dependencias con pip.
# 3. Ejecuta el comando: streamlit run app.py


