import numpy as np              
import matplotlib
import matplotlib.pyplot as plt           
import pandas as pd       
import sklearn.datasets         
import sklearn.preprocessing    
from sklearn import preprocessing


# Ejemplo didáctico: SVD de una matriz 3x2

print("=== Ejemplo didáctico: matriz 3x2 ===\n")
A = np.array([[2, -1],
              [0,  3],
              [1,  1]])
print("Matriz A:\n", A)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("\nMatriz U:\n", U)
print("\nValores singulares (S):\n", S)
print("\nMatriz Vt:\n", Vt)

# Reconstrucción de A

Sigma = np.diag(S)  # Matriz diagonal de valores singulares
matriz_recontruida = U @ Sigma @ Vt
print("\nReconstrucción de A con UΣVt:\n", np.round(matriz_recontruida).astype(int))



#Carga y exploración del dataset de Cáncer de Mama

print("\n=== Carga y exploración del dataset Breast Cancer ===\n")
cancer = sklearn.datasets.load_breast_cancer()
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)

print("Conjunto de datos Cáncer de Mama tiene {} filas y {} columnas\n".format(*df_cancer.shape))
print("Primeras 5 filas:\n", df_cancer.head())
print("\nEstadísticas descriptivas:\n", df_cancer.describe())

# Diccionario de traducción inglés → español
traducciones = {
    'mean radius': 'radio promedio',
    'mean texture': 'textura promedio',
    'mean perimeter': 'perímetro promedio',
    'mean area': 'área promedio',
    'mean smoothness': 'suavidad promedio',
    'mean compactness': 'compacidad promedio',
    'mean concavity': 'concavidad promedio',
    'mean concave points': 'puntos cóncavos promedio',
    'mean symmetry': 'simetría promedio',
    'mean fractal dimension': 'dimensión fractal promedio',
    'radius error': 'error de radio',
    'texture error': 'error de textura',
    'perimeter error': 'error de perímetro',
    'area error': 'error de área',
    'smoothness error': 'error de suavidad',
    'compactness error': 'error de compacidad',
    'concavity error': 'error de concavidad',
    'concave points error': 'error de puntos cóncavos',
    'symmetry error': 'error de simetría',
    'fractal dimension error': 'error de dimensión fractal',
    'worst radius': 'radio peor caso',
    'worst texture': 'textura peor caso',
    'worst perimeter': 'perímetro peor caso',
    'worst area': 'área peor caso',
    'worst smoothness': 'suavidad peor caso',
    'worst compactness': 'compacidad peor caso',
    'worst concavity': 'concavidad peor caso',
    'worst concave points': 'puntos cóncavos peor caso',
    'worst symmetry': 'simetría peor caso',
    'worst fractal dimension': 'dimensión fractal peor caso'
}

# Aplicar traducciones a las columnas
df_cancer.rename(columns=traducciones, inplace=True)




#? Aplicación de la SVD al dataset

#? Estandarización de los Datos

print("\n=== Aplicación de SVD al dataset ===\n")
scaler = sklearn.preprocessing.StandardScaler()
X_std = scaler.fit_transform(df_cancer)


#? Aplicación de la SVD | Descomposición en Valores Singulares

U_cancer, S_cancer, Vt_cancer = np.linalg.svd(X_std, full_matrices=False)

print("Dimensiones de U:", U_cancer.shape)
print("Dimensiones de S:", S_cancer.shape)
print("Dimensiones de Vt:", Vt_cancer.shape)
print("\nPrimeros 5 valores singulares:\n", S_cancer[:5])


#? Visualización inicial de las componentes  

#Calcular los componentes principales (PC)
#Matemáticamente, la proyección es X_estandarizada @ Vt_transpuesta.
# En este caso, Vt ya está transpuesta por la función SVD.
# La matriz de componentes principales es (569, 30)

X_pca = X_std @ Vt_cancer.T

print("\n=== Visualización inicial de 10 componentes ===\n")
component_names = [f"PC{i}" for i in range(1,11)]
X_pca_df = pd.DataFrame(X_pca[:, :10], columns=component_names)


print("Dimension de los datos proyectados:", X_pca_df.shape)
print("\nPrimeras 5 observaciones proyectadas en los 10 componentes principales:\n")
print(X_pca_df.head())


#? Interpretación de componentes con Vt

print("\n=== Interpretación de los componentes mediante Vt ===\n")
Vt_df = pd.DataFrame(Vt_cancer[:10, :],
                     index=[f"PC{i}" for i in range(1, 10+1)],
                     columns=df_cancer.columns)

#obtener los pesos absolutos y ordenarlos
#¿Cuáles de mis 30 variables originales son las más importantes para explicar la varianza en mis datos?
Vt_df1 = pd.DataFrame(Vt_cancer,
                     index=[f"PC{i}" for i in range(1, Vt_cancer.shape[0] + 1)],
                     columns=df_cancer.columns)

print("--- Variables clave que definen el PC1 ---")
# Ordena los pesos de la fila PC1 por magnitud y selecciona los 5 primeros
pc1_loadings = Vt_df.loc['PC1'].abs().sort_values(ascending=False).head(5)
print(pc1_loadings)

print("\n--- Variables clave que definen el PC2 ---")
# Ordena los pesos de la fila PC2 por magnitud y selecciona los 5 primeros
pc2_loadings = Vt_df.loc['PC2'].abs().sort_values(ascending=False).head(5)
print(pc2_loadings)

print("\n--- Variables clave que definen el PC3 ---")
# Ordena los pesos de la fila PC3 por magnitud y selecciona los 5 primeros
pc3_loadings = Vt_df.loc['PC3'].abs().sort_values(ascending=False).head(5)
print(pc3_loadings)




print("Dimensiones de Vt:", Vt_cancer.shape)
print("\nCargas (pesos) de las variables en los primeros 10 componentes:\n")
print(Vt_df)


#? Varianza explicada y acumulada

print("\n=== Varianza explicada ===\n")
varianza_explicada = (S_cancer**2) / np.sum(S_cancer**2)
for i, ratio in enumerate(varianza_explicada[:10]):
    print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%) ")


num_sv_cancer = np.arange(1, S_cancer.size + 1)
cum_var_explained_cancer = [
    np.sum(np.square(S_cancer[:n])) / np.sum(np.square(S_cancer))
    for n in num_sv_cancer
]

print("Varianza acumulada con los primeros 10 componentes (%):\n",
      np.round(np.array(cum_var_explained_cancer[:10]) * 100, 2))


#? Gráfico combinado: valores singulares y varianza explicada

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Línea: varianza acumulada
ax.plot(
    num_sv_cancer,
    cum_var_explained_cancer,
    color="#2171b5",
    label="Varianza explicada (acumulada)",
    alpha=0.8,
    linewidth=2,
    zorder=1000
)

# Puntos: valores singulares normalizados
ax.scatter(
    num_sv_cancer,
    preprocessing.normalize(S_cancer.reshape((1, -1))).flatten(),
    color="#fc4e2a",
    label="Valores singulares (normalizados)",
    alpha=0.8,
    s=80,
    edgecolors="black",
    zorder=1001
)

# Línea discontinua: Scree Plot
ax.plot(
    num_sv_cancer,
    S_cancer / S_cancer.max(),
    color="#33a02c",
    linestyle="--",
    marker="o",
    linewidth=1.5,
    label="Scree plot (valores singulares)",
    zorder=999
)

# Etiquetas en los primeros 10 PCs
for i, (x, y) in enumerate(zip(num_sv_cancer[:10],
                               preprocessing.normalize(S_cancer.reshape((1, -1))).flatten()[:10])):
    ax.text(x, y + 0.03, f"PC{i+1}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

# Personalización del gráfico
ax.set_xticks(num_sv_cancer[:15])
ax.set_xlabel("Número de valores singulares utilizados", fontsize=11)
ax.set_ylabel("Varianza explicada en los datos", fontsize=11)
ax.set_title("Cáncer de Mama: varianza acumulada, valores singulares y scree plot",
             fontsize=14, y=1.03)

ax.set_facecolor("0.98")
ax.legend(loc="best", scatterpoints=1, fontsize=9)
ax.grid(alpha=0.6, linestyle="--", zorder=1)

plt.tight_layout()
plt.show()


#? Visualización de la clasificación con PCA


etiquetas_clase = cancer.target
nombres_clase = cancer.target_names
X_pca_benigno = X_pca[etiquetas_clase == 1]
X_pca_maligno = X_pca[etiquetas_clase == 0]

# Configuración del gráfico
plt.figure(figsize=(10, 8))

# Graficar los dos primeros componentes principales (PC1 y PC2)
# y colorear los puntos según la etiqueta de clase
plt.scatter(
    X_pca_benigno[:, 0],
    X_pca_benigno[:, 1],
    c='green',
    label=nombres_clase[1],
    alpha=0.8
)

plt.scatter(
    X_pca_maligno[:, 0],
    X_pca_maligno[:, 1],
    c='purple',
    label=nombres_clase[0],
    alpha=0.8
)


#? Descripciones de los componentes principales
pc1_desc = "tamaño/forma"
pc2_desc = "textura/rugosidad"


# Añadir las etiquetas y el título
plt.xlabel(f'Primer Componente Principal (PC1) - Define: ({pc1_desc}) - Varianza: {varianza_explicada[0]:.2%}')
plt.ylabel(f'Segundo Componente Principal (PC2)- Define: ({pc2_desc}) - Varianza: {varianza_explicada[1]:.2%}')
plt.title('Clasificación de Tumores (Benignos vs. Malignos) con PCA', fontsize=16)


plt.legend(title="Tipo de Tumor")

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()