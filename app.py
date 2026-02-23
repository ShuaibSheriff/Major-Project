from flask import Flask, render_template, request, redirect, url_for
from markupsafe import Markup
import pickle
import numpy as np
from flask import jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import sqlite3
from model import predict_image
import content

load_dotenv()

app = Flask(__name__, static_folder="")

MODEL_PATH = "model1-randomforest.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model_rf = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model_rf = None

def load_model(modelfile):
    try:
        with open(modelfile, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        raise
def create_db():
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS input_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph_value REAL,
            rainfall REAL,
            predicted_crop TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Call the function to create the database and table
create_db()

# Define the home route
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST', 'GET'])
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        features = [float(request.form['Nitrogen_value']),
                    float(request.form['Phosphorus_value']),
                    float(request.form['Potassium_value']),
                    float(request.form['Temperature_value']),
                    float(request.form['Humidity_value']),
                    float(request.form['Ph_value']),
                    float(request.form['Rainfall_value'])]

        single_pred = np.array(features).reshape(1, -1)

        if model_rf is None:
            return render_template('predict.html', prediction="❌ Model failed to load.")
        
        prediction = model_rf.predict(single_pred)
        predicted_crop = prediction[0]
        store_input_data(features, predicted_crop)
        return render_template('predict.html', prediction=predicted_crop)
    
    return render_template('predict.html')
# filepath: c:\Users\user\Desktop\SHUAIB\AgriPredict\Predict-IT-CropPrediction\app.py
import traceback
...
try:
    with open(MODEL_PATH, 'rb') as f:
        model_rf = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()  # Add this line for detailed error
    model_rf = None
...# filepath: c:\Users\user\Desktop\SHUAIB\AgriPredict\Predict-IT-CropPrediction\app.py
import traceback
...
try:
    with open(MODEL_PATH, 'rb') as f:
        model_rf = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()  # Add this line for detailed error
    model_rf = None
...

# Redirect to the predict.html page when the predict button is clicked
def store_input_data(features, predicted_crop):
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()

    # Insert input values into the table
    cursor.execute('''
        INSERT INTO input_data (nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall, predicted_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', tuple(features + [predicted_crop]))
    
    conn.commit()
    conn.close()


@app.route('/detect', methods=['GET','POST'])
def detect():
    return render_template('detect.html')

import os
from groq import Groq
from model import predict_image
from markupsafe import Markup

# Initialize Groq client with your API key
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            file = request.files.get('img')
            if file:
                img_bytes = file.read()
                
                # Phase 1: Image Prediction (Local Model)
                prediction = predict_image(img_bytes)
                
                # Phase 2: AI Treatment Suggestions (Groq API)
                try:
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert plant pathologist. Provide a concise 3-step treatment plan for the following disease. Use bullet points."
                            },
                            {
                                "role": "user", 
                                "content": f"The plant has been diagnosed with: {prediction}"
                            }
                        ],
                        model="llama-3.3-70b-versatile",
                    )
                    ai_suggestion = chat_completion.choices[0].message.content
                except Exception as groq_err:
                    print(f"Groq API Error: {groq_err}")
                    ai_suggestion = "AI suggestion currently unavailable. Please follow standard agricultural guidelines."

                # Get static info from your local dictionary if available
                res_info = content.disease_dic.get(prediction, "No additional details found.")
                
                return render_template('result.html', 
                                     prediction=prediction, 
                                     info=Markup(res_info),
                                     suggestion=Markup(ai_suggestion.replace('\n', '<br>')))
            
            return render_template('result.html', result="No image uploaded.")
        except Exception as e:
            print(f"Error in processing: {e}")
            return render_template('result.html', result="Error during diagnosis.")


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        input_text = request.form['input_text']

        input_prompt = """You are an expert in understanding soil, agriculture, climate, crops and plant diseases.
        If we provide or ask you anything related to soil or crop or nitrogen, postassium, phosphorous,
        rainfall, soil ph, or humidity or temperature or if any question related to Plant Disease, answer related to that
        else Dont Answer and Say - "I am Optimized only for Agricultural and Plant Disease, Please ask any quereies related to that, Thank You, Team -TechTriad. You have to answer to the question in two or three or four lines and you have to answer only in english and it should be easily understandable."""

        # Handle text input
        response = generate_response(input_text, input_prompt)
        return render_template('chat.html', input_text=input_text, response=response)

    return render_template('chat.html')


def generate_response(input_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, "", prompt])
    return response.text

def get_gemini_response(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_parts = [{
            "mime_type" : uploaded_file.content_type,
            "data" : bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No file Found")



if __name__ == '__main__':
    app.run(debug=True)

