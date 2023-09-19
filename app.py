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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
import os

# Función para obtener datos desde Google Sheets
def get_data_from_gsheets(sheet_url):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_credentials.json', scope)  # Reemplazar por la ruta correcta
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

def main():
    global df_metrics
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
        "B0510 - EXTRACCIÓN DE HULLA (CARBÓN DE PIEDRA)",
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
        "C2229 - FABRICACIÓN DE ARTÍCULOS DE PLÁSTICO N.C.P.",
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
        "C2593 - FABRICACIÓN DE ARTÍCULOS DE CUCHILLERÍA, HERRAMIENTAS DE MANO Y ARTÍCULOS DE FERRETERÍA",
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
        
        # Antes de usar df_metrics, verifica que no sea None y sea un DataFrame válido
if df_metrics is not None and isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:
    	features = st.multiselect('Selecciona características', df_metrics.columns[:-1], default=df_metrics.columns[:-1])
    	if not features:
        	st.warning("Por favor, selecciona al menos una característica para continuar.")
    	else:
        	# Asegurar la definición de scaled_data
        	scaled_data_feature_selected = StandardScaler().fit_transform(df_metrics[features])
        	df_metrics['cluster'] = kmeans.fit_predict(scaled_data_feature_selected)
else:
    	st.warning("No se pudo cargar el DataFrame df_metrics correctamente.")

	    
num_clusters = st.slider("Selecciona el número de clusters", 2, 10, 3)
df_metrics = data[metrics].dropna()  # Eliminar filas con NaN
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

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
df_pca = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
df_pca['cluster'] = df_metrics['cluster']

# Visualización de clusters
st.subheader('Visualización de Clusters')
        
chart = alt.Chart(df_pca).mark_circle(size=60).encode(
        x='PC 1',
        y='PC 2',
        color='cluster:O',
        tooltip=['PC 1', 'PC 2', 'cluster']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Enlace para descargar el dataset con clusters
if st.button('Descargar datos con clusters'):
	st.markdown(download_link_csv(df_metrics, 'data_with_clusters.csv', 'Click aquí para descargar los datos con clusters!'), unsafe_allow_html=True)

        # Método del codo para determinar el número óptimo de clusters
        st.subheader('Determinación del número óptimo de clusters: Método del Codo')

        def elbow_method(data, max_clusters=15):
            distortions = []
            K = range(1, max_clusters)
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(data)
                distortions.append(kmeanModel.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(K, distortions, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Distorsión')
            plt.title('Método del Codo para determinar k óptimo')
            st.pyplot()

        # Selección del número de clusters
        num_clusters = st.slider('Selecciona el número de clusters', 1, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters)
        df_metrics['cluster'] = kmeans.fit_predict(scaled_data)

        # Centroides para la visualización
        cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

        # Actualizar el dataframe PCA con los clusters
        df_pca['cluster'] = df_metrics['cluster']

        # Enlace para descargar el dataset con clusters
        if st.button('Descargar datos con clusters'):
            st.markdown(download_link_csv(df_metrics, 'data_with_clusters.csv', 'Click aquí para descargar los datos con clusters!'), unsafe_allow_html=True)

        # Silhouette Score
        st.subheader('Silhouette Score por Cluster')
        silhouette_scores = silhouette_samples(scaled_data, df_metrics['cluster'])
        df_metrics['silhouette_score'] = silhouette_scores

        # Visualizar silhouette score
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_metrics['cluster'], y=df_metrics['silhouette_score'])
        plt.title('Silhouette Score por Cluster')
        st.pyplot()

# Seleccionar características para análisis
st.subheader('Selecciona características para análisis')
features = st.multiselect('Selecciona características', df_metrics.columns[:-1], default=df_metrics.columns[:-1])
if not features:
    	st.warning("Por favor, selecciona al menos una característica para continuar.")
else:
	# Asegurar la definición de scaled_data
    	scaled_data_feature_selected = StandardScaler().fit_transform(df_metrics[features])
    	df_metrics['cluster'] = kmeans.fit_predict(scaled_data_feature_selected)
        
# Definir num_clusters fuera del bloque condicional
num_clusters = 3  # Puedes establecer un valor predeterminado

# Información detallada del cluster seleccionado
selected_cluster = st.selectbox('Selecciona un cluster para ver detalles', list(range(num_clusters)))
st.write(df_metrics[df_metrics['cluster'] == selected_cluster].describe())

# Mostrar registros del cluster seleccionado
st.subheader(f'Registros del Cluster {selected_cluster}')
num_records = st.slider("Selecciona el número de registros a visualizar", 1, 50, 10)
st.write(df_metrics[df_metrics['cluster'] == selected_cluster].head(num_records))

# Determinar el número óptimo de clusters usando el método del codo
st.subheader('Determinar el Número Óptimo de Clusters')
show_elbow = st.checkbox('Mostrar gráfico del método del codo')
if show_elbow:
    distortions = []
    K = range(1, 15)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(scaled_data_feature_selected)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distorsión')
    plt.title('Método del Codo para determinar k óptimo')
    st.pyplot()

# Dejar al usuario seleccionar el número de clusters
num_clusters = st.slider('Selecciona el número de clusters', 1, 15, 5)
kmeans = KMeans(n_clusters=num_clusters)
df_metrics['cluster'] = kmeans.fit_predict(scaled_data_feature_selected)

# Mostrar la distribución de registros por cluster
st.subheader('Distribución de Registros por Cluster')
st.bar_chart(df_metrics['cluster'].value_counts())

# Seleccionar un cluster para mostrar sus registros
cluster_options = list(range(num_clusters))
selected_cluster = st.selectbox('Selecciona un cluster para visualizar', cluster_options)
st.write(df_metrics[df_metrics['cluster'] == selected_cluster].head(num_records))

# Estadísticas descriptivas por cluster
st.subheader('Estadísticas Descriptivas por Cluster')
show_statistics = st.checkbox('Mostrar estadísticas descriptivas')

if show_statistics:
    cluster_selection = st.selectbox('Elige un cluster para ver sus estadísticas:', range(num_clusters))
    st.write(df_metrics[df_metrics['cluster'] == cluster_selection].describe())
	
# Histograma por Característica y Cluster
st.subheader('Histograma por Característica y Cluster')
show_histogram = st.checkbox('Mostrar histograma')

if show_histogram:
    feature_selection = st.selectbox('Elige una característica para ver su histograma:', df.columns[:-1])  # Excluimos la columna 'cluster'
    bins = st.slider('Selecciona el número de bins:', 5, 100, 20)

    for cluster_id in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_id][feature_selection]
        st.histplot(cluster_data, bins=bins, kde=True, label=f'Cluster {cluster_id}')

    st.legend()
    st.xlabel(feature_selection)
    st.ylabel('Frecuencia')
    st.title(f'Histograma de {feature_selection} por Cluster')

 # Gráfico de Dispersión 2D por Características y Cluster
st.subheader('Gráfico de Dispersión 2D por Características y Cluster')
show_scatter = st.checkbox('Mostrar gráfico de dispersión')

if show_scatter:
    feature_x = st.selectbox('Elige una característica para el eje X:', df.columns[:-1])  # Excluimos la columna 'cluster'
    feature_y = st.selectbox('Elige una característica para el eje Y:', df.columns[:-1])

    fig, ax = plt.subplots()
    for cluster_id in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        ax.scatter(cluster_data[feature_x], cluster_data[feature_y], label=f'Cluster {cluster_id}', alpha=0.7)

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()
    st.pyplot(fig)

    # Visualización de gráfico de dispersión 3D por características y cluster
    st.subheader('Gráfico de dispersión 3D por Características y Cluster')
    show_scatter_3d = st.checkbox('Mostrar gráfico de dispersión 3D')

    if show_scatter_3d:
        feature_x = st.selectbox('Elige una característica para el eje X (3D):', df.columns[:-1])  # Excluimos la columna 'cluster'
        feature_y = st.selectbox('Elige una característica para el eje Y (3D):', df.columns[:-1])
        feature_z = st.selectbox('Elige una característica para el eje Z (3D):', df.columns[:-1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(num_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            ax.scatter(cluster_data[feature_x], cluster_data[feature_y], cluster_data[feature_z], label=f'Cluster {cluster_id}', alpha=0.7)

        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_zlabel(feature_z)
        ax.legend()
        st.pyplot(fig)
    

    # Resumen estadístico de cada cluster
    st.subheader('Resumen estadístico de cada cluster')
    show_summary = st.checkbox('Mostrar resumen')

    if show_summary:
        for cluster_id in range(num_clusters):
            st.write(f'**Cluster {cluster_id}**')
            cluster_data = df[df['cluster'] == cluster_id]
            st.write(cluster_data.describe())


    # Guardar los datos con las etiquetas de cluster
    st.subheader('Guardar datos con etiquetas de cluster')
    save_data = st.button('Guardar datos en CSV')

    if save_data:
        df.to_csv('data_with_clusters.csv')
        st.success('Datos guardados en data_with_clusters.csv')

    # Visualizar distribuciones de características por cluster
    st.subheader('Distribuciones de características por cluster')
    feature_to_view = st.selectbox("Selecciona una característica para visualizar", df.columns[:-1]) # Excluimos la columna 'cluster'
    show_distribution = st.checkbox('Mostrar distribución por cluster')

    if show_distribution:
        for i in cluster_options:
            cluster_data = df[df['cluster'] == i]
            sns.kdeplot(cluster_data[feature_to_view], label=f"Cluster {i}", shade=True)

        plt.xlabel(feature_to_view)
        plt.ylabel('Densidad')
        plt.title(f'Distribución de {feature_to_view} por cluster')
        st.pyplot()

    # Matriz de correlación
    st.subheader('Matriz de correlación de características')
    show_corr = st.checkbox('Mostrar matriz de correlación')

    if show_corr:
        corr_matrix = df.iloc[:, :-1].corr()  # Excluimos la columna 'cluster'
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # para mostrar solo la parte inferior
        cmap = sns.diverging_palette(230, 20, as_cmap=True)  # paleta de colores

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

        st.pyplot(fig)

    # Exploración de clusters específicos
    st.subheader('Explorar datos en un cluster específico')
    cluster_to_explore = st.selectbox("Selecciona un cluster para explorar", cluster_options)
    explore = st.checkbox('Explorar datos')

    if explore:
        cluster_data = df[df['cluster'] == cluster_to_explore]
        st.write(cluster_data)

    # Distribución de características en un cluster específico
    st.subheader('Distribución de características en un cluster específico')
    feature_to_explore = st.selectbox("Selecciona una característica para visualizar", df.columns[:-1])
    show_distribution = st.checkbox('Mostrar distribución')

    if show_distribution:
        cluster_data = df[df['cluster'] == cluster_to_explore]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(cluster_data[feature_to_explore], kde=True, bins=30)
        st.write(f"Distribución de {feature_to_explore} en el cluster {cluster_to_explore}")
        st.pyplot(fig)

    # Comparación de la distribución de una característica entre clusters
    st.subheader('Comparación de la distribución de una característica entre clusters')
    feature_to_compare = st.selectbox("Selecciona una característica para comparar entre clusters", df.columns[:-1])
    show_comparison = st.checkbox('Mostrar comparación')

    if show_comparison:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(x='cluster', y=feature_to_compare, data=df)
        st.write(f"Comparación de {feature_to_compare} entre clusters")
        st.pyplot(fig)


    # Visualizar distribución de características por clusters
    st.subheader('Distribución de características por clusters')
    feature_selection = st.selectbox('Elige una característica para visualizar', df.columns[:-1])  # Excluyendo la columna 'cluster'

    if feature_selection:
        for cluster_num in sorted(df['cluster'].unique()):
            subset = df[df['cluster'] == cluster_num]
            sns.kdeplot(subset[feature_selection], label=f'Cluster {cluster_num}', shade=True)
        
        st.write(f"Distribución de '{feature_selection}' por clusters")
        plt.legend()
        st.pyplot()

    # Cuadro Resumen
    st.subheader('Resumen de la Clustering')

    # Total de registros
    st.write(f'**Total de registros:** {df.shape[0]}')

    # Número de clusters
    st.write(f'**Número de clusters:** {num_clusters}')

    # Registros por cluster
    cluster_counts = df['cluster'].value_counts()
    for cluster, count in cluster_counts.iteritems():
        st.write(f'**Cluster {cluster}:** {count} registros')

	# Final de la aplicación
        st.write("Gracias por usar la aplicación. Si tienes más preguntas o comentarios, ¡no dudes en compartirlos!")

if __name__ == '__main__':
    main()
