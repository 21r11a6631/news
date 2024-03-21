# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
rfc = pickle.load(open('model_news.pkl', 'rb'))
vectorization = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define helper functions
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Define manual testing function
def manual_testing(news):
    # Preprocess the news text
    news = word_drop(news)
    # Transform the news text using the pre-fitted TF-IDF vectorizer
    new_xv_test = vectorization.transform([news])
    # Make prediction using the pre-trained RFC model
    pred_rfc = rfc.predict(new_xv_test)
    return output_label(pred_rfc[0])

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        prediction = manual_testing(news)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
