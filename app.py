from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
filename = 'svm_model.pkl'
model = pickle.load(open(filename, 'rb')) 

# Define the possible race/ethnicity labels
labels = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  
def predict():
    # Get the user input from the web form
    math = int(request.form['math'])
    reading = int(request.form['reading'])
    writing = int(request.form['writing'])

    # Create a data frame with the user input
    data = pd.DataFrame([[math, reading, writing]], columns=['math', 'reading', 'writing'])

    # Make a prediction using the model
    prediction = model.predict(data)

    # Get the predicted label
    result = labels[prediction[0]]

    # Render a new web page with the prediction
    return render_template('index.html', predict=result)
    
if __name__ == '__main__':
    app.run(debug=True)
