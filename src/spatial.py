#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:25:13 2025

@author: user
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from IPython.display import display, HTML
from shapely import wkt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from splot.esda import plot_moran
from pysal.explore import esda
from pysal.lib import weights

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Load data
data = gpd.read_file("/Users/user/projects/projects/EduSeg/data/loc2.geojson")

print(data.head())

data['r1_caminable']=pd.to_numeric(data['r1_caminable'], errors='raise')
data['r1_bicicleta']=pd.to_numeric(data['r1_bicicleta'], errors='raise')
data['r1_transporte']=pd.to_numeric(data['r1_transporte'], errors='raise')

gdf = data.set_crs('EPSG:25830', allow_override=True)

# Map colegios
fig, ax = plt.subplots(figsize=(20, 20))

gdf.plot(ax=ax, color='blue', markersize=10, alpha=0.7)

# Agregar el mapa base
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title('Mapa de Centros Educativos')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
plt.show()


summary = data.describe()
# Exportar como archivo HTML
summary.to_html("summary.html", index=False)

# Map r1
import matplotlib.colors as mcolors

variables = ['r1_caminable', 'r1_bicicleta', 'r1_transporte']

# Asegurar que las columnas sean numéricas
for col in variables:
    gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

# Rango común
vmin = gdf[variables].min().min()
vmax = gdf[variables].max().max()

# Colormap y normalización con centro en 1
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom',
    ['#ffffb2', '#fd8d3c', '#bd0026'],
    N=256
)

norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)

# Crear figura
fig, ax = plt.subplots(1, 3, figsize=(30, 10))

for i, var in enumerate(variables):
    gdf.plot(
        column=var,
        cmap=custom_cmap,
        edgecolor="k",
        linewidth=0.1,
        norm=norm,
        ax=ax[i],
        legend=False
    )
    ax[i].set_title(f'Distribución: {var.replace("r1_", "").capitalize()}')

# Ajustar espacio para dejar hueco abajo
plt.subplots_adjust(bottom=0.15)

# Agregar barra de color debajo de todo
cbar_ax = fig.add_axes([0.2, 0.07, 0.6, 0.03])  # [left, bottom, width, height]
sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm._A = []

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Presión de Demanda')
cbar.ax.tick_params(labelsize=12)

plt.show()

# Spatal Weights


w = weights.distance.Kernel.from_dataframe(
    gdf, fixed=False, k=16)

# Moran 
# Calcular indicador Estandarizado y Lag Estandarizado

gdf["r1_caminable_std"] = gdf["r1_caminable"] - gdf["r1_caminable"].mean()
gdf["r1_caminable_lag_std"] = weights.lag_spatial(w, gdf["r1_caminable_std"])
mi_caminable = esda.moran.Moran(gdf["r1_caminable"], w)

resume_moran = pd.DataFrame({
    "Moran´s I" : [mi_caminable.I],
    "p-value" : [mi_caminable.p_sim.mean()],
    "Z-score" : [mi_caminable.z_sim]
    })
print(f"Moran´s I statistics\n")
display(resume_moran)

fig, ax = plt.subplots(figsize = (10,10))
sns.regplot(
    x = 'r1_caminable_std',
    y = 'r1_caminable_lag_std',
    ci = None,
    data = gdf,
    line_kws={"color": "r"},
    ax=ax)
plt.axvline(0, c="k", alpha=0.5)
plt.axhline(0, c = "k", alpha = 0.5)

plt.text(0.01, 2.5,
         f"Moran´s I: {mi_caminable.I:,.2f}\n"
         f"P-value: {mi_caminable.p_sim.mean()}", size = 20, c='grey')

plt.title("Moran Plot: Caminable", size = 25)

plt.tight_layout()
plt.show()

from esda import geary

# Calcular el índice de Geary
geary_index = geary.Geary(gdf['r1_caminable'], w)

resume_geary = pd.DataFrame({
    "Geary Statistics": [geary_index.C],
    "p-value" : [geary_index.p_sim.mean()],
    "Z-score" : [geary_index.z_rand]
    })
print(f"Geary statistics\n:")
display(resume_geary)

# análisis de clusters espaciales Getis-Ord
from esda.getisord import G
from libpysal.weights import distance

# Calcular el estadístico 
g = G(gdf['r1_caminable'], w)

resume_G = pd.DataFrame({
    "Estadístico G": [g.G],
    "Valor p": [g.p_sim],
    "Estadística z": [g.z_sim]
})
print(f"HotSpot analysis\n:")
display(resume_G)

# Exportar estadísticas
resultados = pd.DataFrame({
    "Estadístico": ["Moran’s I", "Geary C", "Getis-Ord G"],
    "Valor": [0.13067, 0.882146, 0.159505],
    "p-valor": [0.158, 0.338, 0.028],
    "Z-score": [-0.866925, -1.668734, -0.869035]
})

# Exportar como archivo HTML
resultados.to_html("resultados_espaciales.html", index=False)

