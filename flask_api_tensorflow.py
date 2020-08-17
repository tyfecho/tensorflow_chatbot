import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from flask import Flask, request
from keras.models import load_model
from keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import flask
import pickle


data = pickle.load (open("bin/tensorflow/tensorflow-assistant-data.pkl", "rb"))
words = data['words']
classes = data['classes']

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()

tf.keras.backend.set_session(sess)
model = load_model('bin/tensorflow/tensorflow-assistant-model.h5')



def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    
    # bag of words
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# RETURN bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" %w)
    
    return (np.array(bag))


def chatbot_reader(sentence):
    ERROR_THRESHOLD = 0.25

    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
        
    # return tuple of intent and probability
    #reponse = jsonify(return_list)
    return return_list

# request model prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        sentence = request.args['a']
        b = request.args['b']
    elif request.method == 'POST':
        a = request.form['a']
        b = request.form['b']

    # Required because of a bug in Keras when using tensorflow graph cross threads
    with graph.as_default():
        tf.keras.backend.set_session(sess)
        #input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
        #result = model.predict([input_data])[0]

        data = chatbot_reader(sentence.replace("_", " "))
        return flask.jsonify(data)

# start Flask server
app.run(host='0.0.0.0', port=5000, debug=False)