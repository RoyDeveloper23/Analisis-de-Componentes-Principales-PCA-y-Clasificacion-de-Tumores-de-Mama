# 🔬 Clasificador de Tumores de Mama con PCA y Machine Learning

Este proyecto demuestra la efectividad de la Reducción de Dimensionalidad (PCA) para simplificar un dataset complejo (Cáncer de Mama de Scikit-learn) y preparar los datos para una clasificación eficiente.

## 🎯 Objetivos del Proyecto

1.  **Reducción de Dimensionalidad:** Transformar el dataset original de 30 características a un espacio de componentes principales más manejable.
2.  **Interpretación:** Determinar qué variables originales definen a los componentes principales más importantes.
3.  **Visualización:** Demostrar gráficamente la separación de tumores benignos y malignos en solo dos dimensiones (PC1 y PC2).
4.  **Clasificación:** Implementar un modelo de Árbol de Decisión para la predicción del tipo de tumor.

## 📊 Análisis de Componentes Principales (PCA)

El PCA se aplicó sobre los datos estandarizados para identificar las direcciones de máxima varianza.

### Varianza Explicada
Con la selección de **10 componentes principales**, se logra retener aproximadamente el **95%** de la varianza total del dataset, lo cual justifica una drástica reducción de dimensionalidad sin pérdida significativa de información.

### Interpretación de Componentes Clave

| Componente | Varianza Explicada | Variables Clave (Cargas) | Interpretación |
| :---: | :---: | :--- | :--- |
| **PC1** | 44.27% | Puntos Cóncavos, Concavidad, Perímetro | Mide el **Tamaño y la Irregularidad de la Forma** del tumor. |
| **PC2** | 18.97% | Dim. Fractal, Error Dim. Fractal | Mide la **Textura y Rugosidad** de los bordes. |

### Visualización PCA
El gráfico de dispersión (PC1 vs PC2) muestra una clara separación entre los tumores benignos y malignos a lo largo del eje PC1, validando que el PCA conserva la estructura de clasificación.


Tecnologías: 
Python
NumPy
Pandas
Scikit-learn (PCA, SVD)
Matplotlib
Graphviz.


## 💻 Requerimientos e Instalación

Para replicar este proyecto, necesitarás las librerías de Python antes mencionadas, las cuales puedes instalar con el siguiente comando:

pip install numpy pandas scikit-learn matplotlib graphviz
