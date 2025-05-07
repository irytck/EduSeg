#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:48:28 2025

@author: user
"""

# Libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import scale

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Load data
data = gpd.read_file("/Users/user/projects/projects/EduSeg/data/censo_vlc.geojson")

# Rename columns
data = data.rename(columns={
    'namebar' : 'nombre_barrio',
    'namedistri' : 'nombre_distrito',
    'n_vv' : 'numero_viviendas',
    'edad_vv' : 'antiguedad_viviendas',
    'sprf_avg' : 'superficice_media',
    'pct_vvnp' : 'pct_viviendas_no_principal',
    'v_suelo' : 'valor_suelo',
    'v_cnstr_vv' : 'valor_construccion_viviendas',
    'v_total_vv' : 'valor_total_viviendas',
    'v_m2' : 'valor_m2',
    'pct_rented' : 'pct_viviendas_en_alquiler',
    'pct_owned' : 'pct_viviendas_en propiedad',
    'pct_otenur' : 'pct_otro_regimen_de_tenencia',
    'n_pers' : 'numero_personas',
    'edad' : 'edad_media',
    'pct_estsup' : 'pct_personas_con_estudios_superiores',
    'pct_<16' : 'pct_personas_<16_años',
    'pct_>64' : 'pct_personas_>64_años',
    'pct_16_64' : 'pct_personas_16-64_años',
    ' pct_extr' : 'pct_extranjeros',
    'est_avg' : 'estudios_medios',
    'pct_ncdextr' : 'pct_nacidos_extranjeros',
    'pct_ocupds' : 'pct_ocupados',
    'pct_pards' : 'pct_parados',
    'pct_inactv' : 'pct_inactivos',
    'pct_actv' : 'pct_activos',
    'n_extr' : 'numero_extranjeros',
    'pct_ue' : 'pct_UE',
    'pct_na' : 'pct_america_norte',
    'pct_ca' : 'pct_america_central',
    'pct_sa' : 'pct_america_sur',
    'pct_ocean' : 'pct_oceania',
    'n_hogrs' : 'numero_hogares',
    'sup_ocpnt' : 'superficie_por_ocupante',
    'tam_hog_avg' : 'tamaño_medio_hogar',
    'n_actvd' : 'numero_actividades',
    'pob_2011' : 'poblacion_en_2011',
    'pop_rel' : 'crecimiento_poblacion'
    })

data.set_index('CODDISTSEC', inplace=True)

features = ['numero_viviendas', 'antiguedad_viviendas',
'superficice_media', 'pct_viviendas_no_principal', 'valor_suelo',
'valor_construccion_viviendas', 'valor_total_viviendas', 'valor_m2',
'pct_viviendas_en_alquiler', 'pct_viviendas_en propiedad',
'pct_otro_regimen_de_tenencia', 'numero_personas', 'edad_media',
'pct_personas_con_estudios_superiores', 'pct_personas_<16_años',
'pct_personas_>64_años', 'pct_personas_16-64_años', 'pct_extranjeros',
'estudios_medios', 'pct_nacidos_extranjeros', 'pct_ocupados',
'pct_parados', 'pct_inactivos', 'pct_activos', 'numero_extranjeros',
'pct_UE', 'pct_europa', 'pct_africa', 'pct_america_norte',
'pct_america_central', 'pct_america_sur', 'pct_asia', 'pct_oceania',
'numero_hogares', 'superficie_por_ocupante', 'tamaño_medio_hogar',
'gini', 'renta_avg', 'numero_actividades', 'pct_gnd', 'pct_indstr',
'pct_cnstr', 'pct_comserv', 'pct_prof', 'pct_art']


features_scaled = robust_scale(data[features])
features_scaled = pd.DataFrame(features_scaled, columns=features, index=data.index)

# Entrenar modelo PCA
pca_pipe = make_pipeline(PCA(n_components=0.95))
pca_pipe.fit(features_scaled)

modelo=pca_pipe.named_steps['pca'] 

print(f'Number of components retained: {modelo.components_.shape}')

# Varianza explicada por componente
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(modelo.explained_variance_ratio_), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Varianza acumulada explicada por PCA')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.grid(True)
plt.show()

# Importancia de las variables por componente
loading_matrix = pd.DataFrame(modelo.components_.T, 
                              index=features, 
                              columns=[f'PC{i+1}' for i in range(modelo.n_components_)])
# Mostrar las 5 variables que más aportan a la primera componente
loading_matrix['PC1'].abs().sort_values(ascending=False).head(5)

# Proyeccion de los datos en el nuevo espacio
pca_scores = pd.DataFrame(modelo.transform(features_scaled), 
                          columns=[f'PC{i+1}' for i in range(modelo.n_components_)],
                          index=data.index)

# Entrenar PCA con 6 componentes
pca_pipe1 = make_pipeline(PCA(n_components=6))
pca_pipe1.fit(features_scaled)

modelo1 = pca_pipe1.named_steps['pca']

# Varianza explicada por cada componente
print(f'Varianza explicada por cada componente: {modelo1.explained_variance_ratio_}')

# Mostrar la varianza acumulada total
print(f'Varianza explicada acumulada: {modelo1.explained_variance_ratio_.cumsum()}')


# 5 variables más importantes por componente
for i, comp in enumerate(modelo1.components_):
    indices_ordenados = np.argsort(np.abs(comp))[::-1][:5]
    top_vars = [(features_scaled.columns[idx], round(comp[idx], 2)) for idx in indices_ordenados]
    print(f"PC{i+1} - principales variables:")
    for var, peso in top_vars:
        print(f"   {var}: {peso}")
    print()


# Biplot (PC1 vs PC2 con vectores de carga)

def biplot(scores, coeffs, labels=None):
    plt.figure(figsize=(12, 10))
    xs = scores[:, 0]
    ys = scores[:, 1]
    plt.scatter(xs, ys, alpha=0.5)

    for i in range(coeffs.shape[0]):
        plt.arrow(0, 0, coeffs[i, 0]*3, coeffs[i, 1]*3, color='r', alpha=0.5)
        if labels is not None:
            plt.text(coeffs[i, 0]*3.2, coeffs[i, 1]*3.2, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title('Biplot PCA')
    plt.grid()
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.show()

biplot(
    scores=pca_scores.values,
    coeffs=modelo.components_[:2, :].T,
    labels=features
)
