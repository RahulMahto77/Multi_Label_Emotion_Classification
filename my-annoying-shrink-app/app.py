import os
import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import TFBertModel, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

import myfuncs

st.set_page_config(page_title="My Annoying Shrink", layout="centered")

# ---------------- EMOTIONS ---------------- #
GE_taxonomy = [
"admiration","amusement","anger","annoyance","approval","caring",
"confusion","curiosity","desire","disappointment","disapproval",
"disgust","embarrassment","excitement","fear","gratitude","grief",
"joy","love","nervousness","optimism","pride","realization",
"relief","remorse","sadness","surprise","neutral"
]

mapping_emotions = {
  "admiration":["🤩","You always admire what you really don't understand."],
  "amusement":["🥳","The unfortunate who has to travel for amusement lacks capacity for amusement."], 
  "anger":["😡","Bitterness is like cancer. It eats upon the host."], 
  "annoyance":["😤","It could be worse."], 
  "approval":["👍","A man cannot be comfortable without his own approval."], 
  "caring":["😘","Good intentions matter."], 
  "confusion":["🤔","It's ok to be confused."], 
  "curiosity":["🧐","Curiosity drives growth."], 
  "desire":["🤤","Desire shapes direction."], 
  "disappointment":["😞","Things don't always go as planned."], 
  "disapproval":["🙄","Not everything deserves approval."], 
  "disgust":["🤢","Some things are just unpleasant."], 
  "embarrassment":["😳","It happens to everyone."], 
  "excitement":["😆","Excitement brings energy."], 
  "fear":["😱","Fear is often illusion."], 
  "gratitude":["🙏","Gratitude changes mindset."], 
  "grief":["🖤","Time heals gradually."], 
  "joy":["😊","Keep smiling."],
  "love":["😍","Love makes life meaningful."], 
  "nervousness":["😬","Stay calm, breathe."], 
  "optimism":["💪","Hope is powerful."], 
  "pride":["😎","Be proud but humble."], 
  "realization":["🥇","Understanding comes with time."], 
  "relief":["😌","Peace finally."], 
  "remorse":["😔","Learn and move forward."], 
  "sadness":["😢","This too shall pass."], 
  "surprise":["😮","Unexpected things happen."], 
  "neutral":["😐","Just normal state."]
}

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
    model = create_model()

    path = os.path.join(os.path.dirname(__file__), "bert-weights.hdf5")

    if os.path.exists(path):
        model.load_weights(path)
    else:
        st.warning("⚠️ No trained weights found → random predictions")

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
st.title("🧠 My Annoying Shrink")

text = st.text_input("How are you feeling today?")

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
