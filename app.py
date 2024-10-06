from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import pandas as pd

country_data = pd.read_csv('country.csv')
indicator_data = pd.read_csv('indicator.csv')

app = Flask(__name__)

# Load the model
model_pipeline = joblib.load('trained_model.pkl')

country_codes = country_data['code'].unique()
country_names = country_data['country'].unique()
tlas = indicator_data['TLA'].unique()
indicators = indicator_data['indicator'].unique()
issue_tlas = indicator_data['issue_tla'].tolist()  # List of issue_tla values

# Route to render HTML with data for dropdowns
@app.route('/')
def home():
    return render_template('index.html', countries=zip(country_codes, country_names), indicators=zip(zip(tlas,issue_tlas),indicators),)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    print(input_data)
    # Perform prediction
    predicted_value = model_pipeline.predict(input_data)
    return jsonify({'predicted_epi_value': predicted_value[0]})
if __name__ == '__main__':
    app.run(debug=True)
