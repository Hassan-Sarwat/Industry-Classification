import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import joblib

from flask import Flask, jsonify, request

app = Flask(__name__)

#   Read data to fit into vectorizer and encoder

data = pd.read_csv('Job titles and industries.csv')
vectorizer = TfidfVectorizer()
vectorizer.fit(data['job title'])

le = LabelEncoder()
le.fit(data['industry'])


#   load saved model
svm = joblib.load('saved_model.pkl')



@app.route('/<string:job>',methods=['GET'])

#   takes job as parameters, transforms into features
#   predicts based on features and then performs
#   inverse transform on label to get industry

def returnJob(job):
    X = vectorizer.transform([job])
    y = le.inverse_transform(svm.predict(X))
    return jsonify({'Industry: ': y[0]})

if __name__ == '__main__':
    app.run(debug=True)