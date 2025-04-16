import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# Leer el archivo Excel, saltando la primera fila
df = pd.read_csv("data/ganancias.csv", skiprows=1)

# Confirmar columnas
print("Columnas detectadas:", df.columns)

# Transformar de formato ancho a largo
df_largo = df.melt(id_vars="mes", var_name="año", value_name="ganancias")

# Asegurar orden correcto de meses
orden_meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
df_largo["mes"] = pd.Categorical(df_largo["mes"], categories=orden_meses, ordered=True)

# Visualización
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_largo, x="mes", y="ganancias", hue="año", marker="o")
plt.title("Ganancias del 1er Semestre (2021–2023)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/lineas.png", dpi=300)
plt.show()



# Agrupamos por año y mes para tener solo una fila por combinación
df_cluster = df_largo.groupby(["año", "mes"], as_index=False,observed=True)["ganancias"].sum()
print("Datos agrupados:\n", df_cluster)
# Clustering usando solo la columna de ganancias
X = df_cluster[["ganancias"]]

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X)

# Visualización
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_cluster, x="mes", y="ganancias", hue="cluster", palette="Set1")
plt.title("Clustering de Ganancias por Mes (KMeans)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_clustering.png", dpi=300)
plt.show()

# Visualizaciones adicionales del análisis de ganancias

# === 1. BARRAS: Totales anuales ===
plt.figure(figsize=(8, 5))
resumen_anual = df_largo.groupby("año")["ganancias"].sum().sort_index()
sns.barplot(x=resumen_anual.index, y=resumen_anual.values, palette="muted")
plt.title("Total de Ganancias por Año")
plt.xlabel("Año")
plt.ylabel("Suma de Ganancias")
plt.tight_layout()
plt.savefig("output/total_ganancias_anuales.png", dpi=300)
plt.show()

# === 2. HEATMAP: Ganancias por mes-año ===
plt.figure(figsize=(8, 6))
tabla_cruzada = df_largo.pivot(index="mes", columns="año", values="ganancias")
sns.heatmap(tabla_cruzada, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Ganancia"})
plt.title("Mapa de Calor: Ganancias por Mes y Año")
plt.xlabel("Año")
plt.ylabel("Mes")
plt.tight_layout()
plt.savefig("output/mapa_calor_ganancias.png", dpi=300)
plt.show()

# === 3. BOXPLOT: Dispersión anual ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_largo, x="año", y="ganancias", palette="pastel")
plt.title("Dispersión de Ganancias por Año")
plt.xlabel("Año")
plt.ylabel("Ganancia Mensual")
plt.tight_layout()
plt.savefig("output/boxplot_por_anio.png", dpi=300)
plt.show()

# === 4. REGRESIÓN LINEAL: Predicción de tendencia ===
# Preparar datos para regresión
df_largo["año_numerico"] = pd.to_numeric(df_largo["año"])
X_data = df_largo[["año_numerico"]]
y_data = df_largo["ganancias"]

# Entrenamiento del modelo lineal
reg_model = LinearRegression()
reg_model.fit(X_data, y_data)
df_largo["estimado"] = reg_model.predict(X_data)

# Visualización de regresión
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_largo, x="año", y="ganancias", label="Valores observados", color="teal")
sns.lineplot(data=df_largo, x="año", y="estimado", label="Línea de tendencia", color="darkred")
plt.title("Tendencia de Ganancias (Regresión Lineal)")
plt.xlabel("Año")
plt.ylabel("Ganancias")
plt.legend()
plt.tight_layout()
plt.savefig("output/tendencia_regresion_lineal.png", dpi=300)
plt.show()



