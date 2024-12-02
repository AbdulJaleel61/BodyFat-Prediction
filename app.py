from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the saved hybrid model
with open('hybrid_model.pkl', 'rb') as file:
    hybrid_model = pickle.load(file)

# Extract individual models from the hybrid model
decision_tree = hybrid_model['decision_tree']
random_forest = hybrid_model['random_forest']
mlp = hybrid_model['mlp']

# Initialize Flask app
app = Flask(__name__)

# Home route to render input form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route to handle form data and render result
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    chest = float(request.form['Chest'])
    hip = float(request.form['Hip'])
    weight = float(request.form['Weight'])
    abdomen = float(request.form['Abdomen'])
    density = float(request.form['Density'])
    
    # Prepare data for prediction
    input_features = np.array([[chest, hip, weight, abdomen, density]])
    
    # Make predictions with individual models
    dt_prediction = decision_tree.predict(input_features)
    rf_prediction = random_forest.predict(input_features)
    mlp_prediction = mlp.predict(input_features)

    # Combine predictions (simple averaging)
    hybrid_prediction = (dt_prediction + rf_prediction + mlp_prediction) / 3

    # Render result.html with prediction
    return render_template('result.html', body_fat_prediction =float(hybrid_prediction[0]))


# Home route to render result form
@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/About Us')
def About():
    return render_template('About_Us.html')

@app.route('/Contact')
def Contact():
    return render_template('Contact_Us.html')

if __name__ == '__main__':
    app.run(debug=True)

