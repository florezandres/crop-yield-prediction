from data_processing import cargar_y_preparar_datos
from data_analysis import rendimiento_por_departamento, matriz_correlacion, tabla_contingencia
from model_training import entrenar_modelo
from prediction import predecir_produccion
import pickle

url = "https://www.datos.gov.co/api/views/2pnw-mmge/rows.csv?accessType=DOWNLOAD"

# 1. Cargar y limpiar datos
data = cargar_y_preparar_datos(url)

# 2. Analítica de datos
#rendimiento_por_departamento(data)
#matriz_correlacion(data)
#tabla_contingencia(data)

# 3. Entrenar modelo
model, encoder, cat_features, num_features, features = entrenar_modelo(data)

# Guardar data limpia
with open("data_limpia.pkl", "wb") as f:
    pickle.dump(data, f)

# 4. Predicción ejemplo
resultado = predecir_produccion(model, encoder, cat_features, num_features, features,
                                "AMAZONAS", "LETICIA", "MAIZ", 10)

print(f"Predicción de producción: {resultado:.2f} toneladas")

# Guardar todo en archivos pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump({
        "cat_features": cat_features,
        "num_features": num_features,
        "features": features
    }, f)