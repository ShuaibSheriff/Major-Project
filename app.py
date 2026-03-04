from flask import Flask, render_template, request, redirect, url_for, jsonify
from markupsafe import Markup
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from PIL import Image
import sqlite3
import traceback
from groq import Groq
from model import predict_image

load_dotenv()

app = Flask(__name__, static_folder='static')

MODEL_PATH = "model1-randomforest.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model_rf = pickle.load(f)
    print("✅ Crop model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load crop model: {e}")
    traceback.print_exc()
    model_rf = None

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def create_db():
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()
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

create_db()


def store_input_data(features, predicted_crop):
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO input_data (nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall, predicted_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', tuple(features + [predicted_crop]))
    conn.commit()
    conn.close()


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
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
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return render_template('predict.html', prediction="❌ Error during prediction. Please check your inputs.")

    return render_template('predict.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    return render_template('detect.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            file = request.files.get('img')
            if file:
                img_bytes = file.read()

                # Phase 1: Image Prediction (Local ResNet34 Model)
                prediction = predict_image(img_bytes)

                # Phase 2: AI Treatment Suggestions (Groq API)
                try:
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert plant pathologist and an agricultural advisor. Provide a concise 3-step treatment plan for the following disease. Use bullet points. Ensure your response includes common market names for any chemicals suggested, and strongly emphasize organic and easily accessible household alternatives."
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

                # Get static info from local dictionary if available
                import content
                res_info = content.disease_dic.get(prediction, "No additional details found.")

                return render_template('result.html',
                                       prediction=prediction,
                                       info=Markup(res_info),
                                       suggestion=Markup(ai_suggestion.replace('\n', '<br>')))

            return render_template('result.html', result="No image uploaded.")
        except Exception as e:
            print(f"Error in processing: {e}")
            traceback.print_exc()
            return render_template('result.html', result="Error during diagnosis.")


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        input_text = request.form['input_text']
        language_preference = request.form.get('language', 'en')

        # Map lang code to full name for the prompt
        lang_map = {'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi'}
        target_lang = lang_map.get(language_preference, 'English')

        system_prompt = f"""You are an expert in understanding soil, agriculture, climate, crops and plant diseases.
If we provide or ask you anything related to soil or crop or nitrogen, potassium, phosphorous,
rainfall, soil ph, or humidity or temperature or if any question related to Plant Disease, answer related to that.
Otherwise do not answer and say - "I am Optimized only for Agricultural and Plant Disease, Please ask any queries related to that, Thank You, Team - TechTriad."
Answer in two to four lines and keep it easily understandable.
CRITICAL: You must answer ONLY in the {target_lang} language."""

        response = generate_response(input_text, system_prompt)
        return render_template('chat.html', input_text=input_text, response=response)

    return render_template('chat.html')


def generate_response(input_text, system_prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq chat error: {e}")
        return "Sorry, the AI assistant is currently unavailable. Please try again later."


if __name__ == '__main__':
    app.run(debug=True)
