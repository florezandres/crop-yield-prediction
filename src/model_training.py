import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def entrenar_modelo(data):
    features = ["DEPARTAMENTO","MUNICIPIO", "CULTIVO", "Área Sembrada (ha)"]
    target = "Producción (t)"

    cat_features = ["DEPARTAMENTO","MUNICIPIO", "CULTIVO"]
    num_features = ["Área Sembrada (ha)"]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cat = encoder.fit_transform(data[cat_features])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features))

    X = pd.concat([encoded_df, data[num_features].reset_index(drop=True)], axis=1)
    data['target_log'] = np.log1p(data['Producción (t)'])
    y = data['target_log']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
                             learning_rate=0.1, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}, R2: {r2}")

    return model, encoder, cat_features, num_features, features
