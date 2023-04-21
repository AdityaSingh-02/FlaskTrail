from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
with open('svm_classifier.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template with the input form

@app.route('/predict',methods=['POST'])
def predict():
    # Get the input values from the form
    # pregnancies = float(request.form['pregnancies'])
    # glucose = float(request.form['glucose'])
    # blood_pressure = float(request.form['blood_pressure'])
    # skin_thickness = float(request.form['skin_thickness'])
    # insulin = float(request.form['insulin'])
    # bmi = float(request.form['bmi'])
    # diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    # age = float(request.form['age'])

    # Getting values from URL
    pregnancies = float(request.args.get('pregnancies'));
    glucose = float(request.args.get('glucose'));
    blood_pressure = float(request.args.get('blood_pressure'));
    skin_thickness = float(request.args.get('skin_thickness'));
    insulin = float(request.args.get('insulin'));
    bmi = float(request.args.get('bmi'));
    diabetes_pedigree_function = float(request.args.get('diabetes_pedigree_function'));
    age = float(request.args.get('age'));

    # Create a numpy array with the input values
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Scale the input data using the same StandardScaler object used to train the SVM model
    with open('scaler.pkl', 'rb') as fil:
        scaler = pickle.load(fil)
    standardized_input_data = scaler.transform(input_data)

    # Use the SVM model to predict the outcome (0 or 1)
    prediction = model.predict(standardized_input_data)

    # Render the HTML template with the prediction
    return render_template('result.html', prediction=prediction)

@app.route('/User')
def User():
    username = request.args.get('Username')
    print(username)
    return render_template('result.html', prediction=username)

if __name__ == '__main__':
    app.run(debug=True)
