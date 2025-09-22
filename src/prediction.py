import pandas as pd
import numpy as np

def predecir_produccion(model, encoder, cat_features, num_features, features,
                        departamento, municipio, cultivo, area_sembrada):
    input_data = pd.DataFrame([[departamento, municipio, cultivo, area_sembrada]], columns=features)
    encoded_input = encoder.transform(input_data[cat_features])
    input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_features))
    input_final = pd.concat([input_df, input_data[num_features].reset_index(drop=True)], axis=1)

    prediccion_log = model.predict(input_final)[0]
    prediccion_real = np.expm1(prediccion_log) * 1.5  # Factor de corrección

    return prediccion_real
