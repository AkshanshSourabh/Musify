import requests
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import requests
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import os
import random
import string

with open('data.json') as database:
    data1 = json.load(database)
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"input patterns": inputs, 'tags': tags})
data = data.sample(frac=1)

data['input patterns'] = data['input patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['input patterns'] = data['input patterns'].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['input patterns'])
train = tokenizer.texts_to_sequences(data['input patterns'])

x_train = pad_sequences(train)
le = LabelEncoder()

y_train = le.fit_transform(data['tags'])
input_shape = x_train.shape[1]
print(input_shape)
vocabulary = len(tokenizer.word_index)
print("Number of unique words : ", vocabulary)
output_length = le.classes_.shape[0]
print("Output length : ", output_length)
i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
train = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('model.h5', train)
print("model created")

model = load_model('model.h5')  # load the saved model

import requests
import json
import os

# Load secrets from environment variables
secrets = {
    "spotify": "16329bd87efb4ab48d314dbdda463054"
}

def get_sentiment(text):
    payload = {
        "key": "60fde252181e4de202325b4b9e338706",
        "txt": text
    }
    response = requests.post('https://api.meaningcloud.com/sentiment-2.1', data=payload)
    response_dict = json.loads(response.text)

    if "score_tag" in response_dict:
        sentiment = response_dict["score_tag"]
    else:
        sentiment = "Neutral"

    return sentiment

def get_recommendations(emotion, secrets):
    # Set up the API request
    url = "https://api.spotify.com/v1/tracks"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secrets['spotify']}"
    }
    params = {
        "limit": "4",
        "seed_genres": "pop",
        "target_valence": "0.5"
    }

    # Modify the API request based on the detected emotion
    if emotion == "Positive":
        params["seed_genres"] = "happy"
        params["target_valence"] = "0.8"
    elif emotion == "Negative":
        params["seed_genres"] = "sad"
        params["target_valence"] = "0.2"

    # Send the API request
    response = requests.get(url, headers=headers, params=params)


    data = response.json()

   
    tracks = data['tracks']

    # print the names and artists of the recommended tracks
    for track in tracks:
        print(f"Track: {track['name']}")
        print(f"Artist: {track['artists'][0]['name']}")
    # with open('response.json', 'r') as file:
    #     data = json.load(file)

# Access the track's album name
    # if emotion == "Positive":
    #     print(data['tracks'][0]['album']['name'])
    #     print(data['tracks'][0]['album']['naam'])
    #     print(data['tracks'][0]['album']['daam'])
    #     print(data['tracks'][0]['album']['saam'])
    # elif emotion == "Negative":
    #     print(data['tracks'][0]['album']['seed'])
    #     print(data['tracks'][0]['album']['feed'])
    #     print(data['tracks'][0]['album']['kaam'])
    #     print(data['tracks'][0]['album']['deed'])


# Define the main function for the chatbot
def song_recommendation_chatbot():
    print("Welcome to the Song Recommendation Chatbot!")
    while True:
        # Get user input and analyze the sentiment
        text = input("\nHow are you feeling today? ")
        sentiment = get_sentiment(text)

        # Get song recommendations based on the detected emotion
        if sentiment == "P" or sentiment == "P+":
            emotion = "Positive"
            
        elif sentiment == "N" or sentiment == "N+":
            emotion = "Negative"
        else:
            print("Sorry, I didn't understand your emotion. Please try again.")
            continue

        print("Here are some songs you can listen now according to your mood:\n")
        recommendations = get_recommendations(emotion, secrets)

        # Print the song recommendations
        if recommendations is not None:
            print(f"Here are some song recommendations for your {emotion.lower()} mood:")
            for i, track in enumerate(recommendations):
                print(f"{i+1}. {track}")
        else:
            pass

# Call the main function+
song_recommendation_chatbot()



