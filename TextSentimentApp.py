import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# for crunching github data
import requests
import json
import time
import datetime
import os
import fnmatch
import pandas as pd

# NLP Packages
from textblob import TextBlob
import random
import time

# TensorFlow/Keras (fixed imports)
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.datasets import imdb

# Patch numpy load (to allow pickle=True)
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)


# ---------------------------
# Streamlit App Starts Here
# ---------------------------

st.title("Text Sentiment Classification App")
st.write("Upload a pre-trained Keras model and classify IMDB text data.")


# Use st.cache_resource instead of deprecated st.cache
@st.cache_resource
def load_intermediate():
    # Example: load imdb dataset
    max_features = 5000
    maxlen = 400
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return (X_train, y_train), (X_test, y_test)


uploaded_file = st.file_uploader("Upload a trained Keras (.h5) model", type=["h5"])

if uploaded_file is not None:
    with open("uploaded_model.h5", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Model uploaded successfully!")

    # Load model
    model = keras.models.load_model("uploaded_model.h5")

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_intermediate()

    st.write("### Evaluating Model on Test Data...")
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test score: {score:.4f}")
    st.write(f"Test accuracy: {acc:.4f}")

    # Random prediction
    if st.button("Run Random Prediction"):
        idx = random.randint(0, len(X_test) - 1)
        sample = np.expand_dims(X_test[idx], axis=0)
        pred = model.predict(sample)
        sentiment = "Positive 😀" if pred[0][0] > 0.5 else "Negative 😞"
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"True Label: {'Positive' if y_test[idx] == 1 else 'Negative'}")
# Build a model in code
def build_model(max_features=5000, maxlen=400):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# Train or load dataset
(X_train, y_train), (X_test, y_test) = load_intermediate()

model = build_model()
model.fit(X_train, y_train,
          batch_size=32,
          epochs=2,
          validation_split=0.2,
          verbose=1)

st.success("✅ Model trained inside app!")
score, acc = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Test score: {score:.4f}")
st.write(f"Test accuracy: {acc:.4f}")

