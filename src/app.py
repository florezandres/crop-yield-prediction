from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar modelo y encoder desde pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Cargar data limpia desde pickle
with open("data_limpia.pkl", "rb") as f:
    data = pickle.load(f)

cat_features = metadata["cat_features"]
num_features = metadata["num_features"]
features = metadata["features"]

def calcular_media_produccion(data, municipio, cultivo):
    """
    Devuelve la media de Producción (t) para un municipio y cultivo específicos.
    """
    # Filtrar por municipio y cultivo
    filtrado = data[
        (data["MUNICIPIO"] == municipio) &
        (data["CULTIVO"] == cultivo)
    ]

    # Calcular media de Producción
    if filtrado.shape[0] == 0:
        return None  # No hay datos para ese municipio y cultivo
    else:
        media = filtrado["Producción (t)"].mean()
        return round(media, 2)

def predecir_produccion(departamento, municipio, cultivo, area_sembrada):
    input_data = pd.DataFrame([[departamento, municipio, cultivo, area_sembrada]], columns=features)
    encoded_input = encoder.transform(input_data[cat_features])
    input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_features))
    input_final = pd.concat([input_df, input_data[num_features].reset_index(drop=True)], axis=1)

    prediccion_log = model.predict(input_final)[0]
    prediccion_real = np.expm1(prediccion_log) * 1.5  # Factor de corrección
    return prediccion_real

@app.route('/predict', methods=['GET'])
def predict():
    # Query params}
    departamento = request.args.get('departamento')
    municipio = request.args.get('municipio')
    cultivo = request.args.get('cultivo')
    area_sembrada = float(request.args.get('area_sembrada'))

    calcular_media_produccion(data, municipio, cultivo)

    resultado = predecir_produccion(
        departamento,
        municipio,
        cultivo,
        area_sembrada
    )

    return jsonify({
        "model_prediction":{
            "produccion_estimada": float(round(resultado, 2)),
            "departamento": departamento,
            "municipio": municipio,
            "cultivo": cultivo,
            "area_sembrada": area_sembrada
            },
        "media_produccion": calcular_media_produccion(data, municipio, cultivo)
        })

if __name__ == '__main__':
    app.run(debug=True)
