import pickle

departamento = 'AMAZONAS'
municipio = 'LETICIA'
cultivo = 'MAIZ'
area_sembrada = 10

url = f'http://localhost:5000/predict/{departamento}/{municipio}/{cultivo}/{area_sembrada}'

with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Ejecutar el modelo
variables = [departamento, municipio, cultivo, area_sembrada]
print(model.predict(variables))