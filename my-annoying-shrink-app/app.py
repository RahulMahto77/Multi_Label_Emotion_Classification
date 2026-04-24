import os
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

from transformers import TFBertModel, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

import myfuncs

st.set_page_config(page_title="Emotion Detector", layout="centered")

# ---------------- EMOTIONS ---------------- #
GE_taxonomy = [
"admiration","amusement","anger","annoyance","approval","caring",
"confusion","curiosity","desire","disappointment","disapproval",
"disgust","embarrassment","excitement","fear","gratitude","grief",
"joy","love","nervousness","optimism","pride","realization",
"relief","remorse","sadness","surprise","neutral"
]

mapping_emotions = {
    emo: ["🙂", f"You are feeling {emo}"] for emo in GE_taxonomy
}

# ---------------- DOWNLOAD WEIGHTS ---------------- #
def download_weights():
    url = "https://media.githubusercontent.com/media/RahulMahto77/Multi_Label_Emotion_Classification/main/bert-weights.hdf5"
    path = "bert-weights.hdf5"

    if not os.path.exists(path):
        st.info("⬇️ Downloading model weights (first time only)...")
        try:
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)
            st.success("✅ Weights downloaded")
        except:
            st.warning("⚠️ Could not download weights. Running without them.")

# ---------------- TOKENIZER ---------------- #
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ---------------- MODEL ---------------- #
def create_model():
    bert = TFBertModel.from_pretrained("bert-base-uncased")

    input_ids = Input(shape=(48,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(48,), dtype=tf.int32, name="attention_mask")

    outputs = bert(input_ids, attention_mask=attention_mask)[1]

    x = Dropout(0.3)(outputs)
    out = Dense(len(GE_taxonomy), activation="sigmoid",
                kernel_initializer=TruncatedNormal(stddev=0.02))(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=out)
    return model

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    download_weights()

    model = create_model()

    if os.path.exists("bert-weights.hdf5"):
        model.load_weights("bert-weights.hdf5")
        st.success("✅ Model loaded with trained weights")
    else:
        st.warning("⚠️ Running with base BERT (random predictions)")

    return model

model = load_model()

# ---------------- TOKENIZE ---------------- #
def tokenize(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="tf"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }

# ---------------- PREDICT ---------------- #
def predict(text):
    text = myfuncs.preprocess_corpus(text)
    inputs = tokenize(text)

    probs = model.predict(inputs, verbose=0)[0]
    best = GE_taxonomy[int(np.argmax(probs))]

    return probs, best

# ---------------- UI ---------------- #
st.title("🧠 Emotion Detector")

text = st.text_input("Enter your text")

if text:
    probs, best = predict(text)

    st.subheader("Detected Emotion:")
    st.write(best)

    st.subheader("Advice:")
    st.write(mapping_emotions[best][1])

    if st.button("Show probabilities"):
        for i, p in enumerate(probs):
            if p > 0.3:
                st.write(f"{GE_taxonomy[i]} → {p:.2f}")
