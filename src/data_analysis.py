import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def rendimiento_por_departamento(data):
    rendimiento_departamento = data.groupby('DEPARTAMENTO')['Rendimiento (t/ha)'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rendimiento_departamento.values, y=rendimiento_departamento.index)
    plt.title('Rendimiento promedio por departamento')
    plt.show()

def matriz_correlacion(data):
    plt.figure(figsize=(10, 6))
    corr = data[['Área Sembrada (ha)', 'Área Cosechada (ha)', 'Producción (t)', 'Rendimiento (t/ha)']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de correlación entre variables numéricas')
    plt.show()

def tabla_contingencia(data):
    contingencia = pd.crosstab(data['DEPARTAMENTO'], data['GRUPO DE CULTIVO'])
    print("Tabla de contingencia departamento vs grupo de cultivo:")
    print(contingencia)
