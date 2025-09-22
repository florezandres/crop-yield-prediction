import pandas as pd
import re

def cargar_y_preparar_datos(url):
    data = pd.read_csv(url)

    # Limpiar nombres de columnas
    new_columns = [re.sub(r'\s+', ' ', col.replace('\n', ' ').strip()) for col in data.columns]
    data.columns = new_columns

    # Rellenar nulos
    data["Rendimiento (t/ha)"] = data["Rendimiento (t/ha)"].fillna(0)
    data["NOMBRE CIENTIFICO"] = data["NOMBRE CIENTIFICO"].fillna("no disponible")

    # Eliminar duplicados y nulos restantes
    data = data.drop_duplicates().dropna()

    return data
