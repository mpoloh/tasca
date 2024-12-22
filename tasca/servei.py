import pickle
import pandas as pd
from flask import Flask, jsonify, request

# Clases de especies de pingüinos
classes = ['Adelie', 'Chinstrap', 'Gentoo']

def predict_single(penguin, model):
    # Convertir los datos del pingüino a un DataFrame
    penguin_df = pd.DataFrame({
        'flipper_length_mm': [penguin['flipper_length_mm']],
        'body_mass_g': [penguin['body_mass_g']],
        'island': [penguin['island']],
        'sex': [penguin['sex']]
    })

    # Realizar la predicción
    y_pred = model.predict(penguin_df)  # Predicción de la clase
    y_prob = model.predict_proba(penguin_df)  # Probabilidades para cada clase

    # Comprobar si hay predicciones
    if len(y_pred) == 0 or len(y_prob) == 0:
        return None, None
    
    # Obtener la probabilidad de la clase predicha
    pred_probability = y_prob[0][classes.index(y_pred[0])]  # Acceder a la probabilidad correspondiente a la clase predicha
    
    return y_pred[0], float(pred_probability)  # Devolver la especie y su probabilidad

def predict(model):
    penguin = request.get_json()
    
    # Asegúrate de que todos los parámetros necesarios están presentes
    if not all(key in penguin for key in ['flipper_length_mm', 'body_mass_g', 'island', 'sex']):
        return jsonify({'error': 'Missing parameters'}), 400
    
    especie, probabilitat = predict_single(penguin, model)
    
    if especie is None:
        return jsonify({'error': 'Prediction failed'}), 500
   
    result = {
        'species': especie,
        'probability': probabilitat
    }
    return jsonify(result)

app = Flask('penguins')

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        model = pickle.load(f)
    return predict(model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        model = pickle.load(f)
    return predict(model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        model = pickle.load(f)
    return predict(model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        model = pickle.load(f)
    return predict(model)

if __name__ == '__main__':
    app.run(debug=True, port=8000)