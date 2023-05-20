from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from model import train_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import spacy
import pickle
import joblib

from text_summarizer import summarize_text
from vid_face_expression_extraction import extract_expression

app = Flask(__name__)

filename = 'trained_model.pkl'
model = pickle.load(open(filename, 'rb'))
pipeline = joblib.load('pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/anomaly', methods=["POST"])
def anomaly():
    user_input = request.form.get('paragraph')
    print(user_input)
    input_stuffs = [user_input]
    nlp = spacy.load("en_core_web_sm")
    string = ' '.join(input_stuffs)
    doc = nlp(string)
    anomaly_results = []  
    normal_results = []
    for sent in doc.sents:
        input = [str(sent)]
        print(input)
        # convert text to feature vectors
        input_data_features = pipeline.transform(input)
        # # making prediction
        prediction = model.predict(input_data_features)
        # If the input is detected to be anomalous
        if (prediction[0]==1):
            normal_results.append(str(sent))
        else:
            anomaly_results.append(str(sent))

    print("Normal: -------------- ",normal_results)
    print("Anomaly: -------------- ",anomaly_results)
    response = {"normal": normal_results, "anomaly": anomaly_results}
    return jsonify(response)

@app.route('/summarize', methods=["POST"])
def summarize():
    summarization_data = request.form.get('summarization')
    summarization_input = [summarization_data]
    summarization = summarize_text(summarization_input)
    return jsonify(summarization)

# Add methods=["POST"] and request.form.get('video)
@app.route('/facial')
def facial():
    # facial_input = request.form.get('video')
    facial_input = 'C:/Users/asus/Pictures/Camera Roll/video1.MOV'
    facial_result = extract_expression(facial_input)
    return jsonify(facial_result)

if __name__ == '__main__':
    app.run(debug=True)