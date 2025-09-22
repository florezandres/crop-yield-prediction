from data_processing import cargar_y_preparar_datos
from data_analysis import rendimiento_por_departamento, matriz_correlacion, tabla_contingencia
from model_training import entrenar_modelo
from prediction import predecir_produccion

url = "https://www.datos.gov.co/api/views/2pnw-mmge/rows.csv?accessType=DOWNLOAD"

# 1. Cargar y limpiar datos
data = cargar_y_preparar_datos(url)

# 2. Analítica de datos
rendimiento_por_departamento(data)
matriz_correlacion(data)
tabla_contingencia(data)

# 3. Entrenar modelo
model, encoder, cat_features, num_features, features = entrenar_modelo(data)

# 4. Predicción ejemplo
resultado = predecir_produccion(model, encoder, cat_features, num_features, features,
                                "AMAZONAS", "LETICIA", "MAIZ", 10)

print(f"Predicción de producción: {resultado:.2f} toneladas")

