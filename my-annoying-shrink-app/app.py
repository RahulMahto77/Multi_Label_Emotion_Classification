# ================= IMPORTS =================
import numpy as np
import streamlit as st
from PIL import Image

from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

import myfuncs


# ================= LABELS =================
GE_taxonomy = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


# ================= EMOJI MAP =================
mapping_emotions = {
    "admiration": ["🤩", "You admire what you don’t understand."],
    "amusement": ["🥳", "Amusement unlocked!"],
    "anger": ["😡", "Anger burns like fire."],
    "annoyance": ["😤", "It could be worse."],
    "approval": ["👍", "Self-approval matters."],
    "caring": ["🥰", "Care is powerful."],
    "confusion": ["🤨", "It's okay to be confused."],
    "curiosity": ["🤔", "Curiosity drives learning."],
    "desire": ["🤤", "Strong desire detected."],
    "disappointment": ["😔", "Expectations not met."],
    "disapproval": ["🙄", "Not impressed."],
    "disgust": ["🤢", "Disgust detected."],
    "embarrassment": ["😳", "Embarrassment moment."],
    "excitement": ["😆", "Excited energy!"],
    "fear": ["😱", "Fear detected."],
    "gratitude": ["🙏", "Be grateful always."],
    "grief": ["🥀", "Loss detected."],
    "joy": ["😊", "Happiness detected."],
    "love": ["😍", "Love is in the air."],
    "nervousness": ["😬", "Feeling nervous."],
    "optimism": ["💪", "Stay positive."],
    "pride": ["😎", "Proud moment."],
    "realization": ["💡", "Realization hit."],
    "relief": ["😮‍💨", "Relief achieved."],
    "remorse": ["😞", "Regret detected."],
    "sadness": ["😭", "Sadness detected."],
    "surprise": ["😮", "Surprise!"],
    "neutral": ["😐", "Neutral state"]
}


# ================= PREDICTION =================
def predict_sample(text, model, tokenizer, threshold=0.85):

    text = myfuncs.preprocess_corpus(text)
    sample = myfuncs.tokenize(tokenizer, text)

    probs = model.predict(sample, verbose=0)[0]

    labels = [1 if p > threshold else 0 for p in probs]

    best_label = GE_taxonomy[int(np.argmax(probs))]

    detected = [GE_taxonomy[i] for i, v in enumerate(labels) if v == 1]
    detected_probs = [p for p in probs if p > threshold]

    if len(detected) == 0:
        detected = ["neutral"]
        detected_probs = ["-"]
        best_label = "neutral"

    return detected, detected_probs, best_label


# ================= MODEL =================
@st.cache_resource
def create_model():

    config = BertConfig.from_pretrained("bert-base-uncased")

    bert = TFBertModel.from_pretrained("bert-base-uncased", config=config)

    input_ids = Input(shape=(48,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(48,), dtype="int32", name="attention_mask")

    bert_output = bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    pooled_output = bert_output.pooler_output

    x = Dropout(0.3)(pooled_output)

    output = Dense(
        27,
        activation="sigmoid",
        kernel_initializer=TruncatedNormal(stddev=config.initializer_range)
    )(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    return model


@st.cache_resource
def load_model(model):
    model.load_weights("bert-weights.hdf5")
    return model


# ================= STREAMLIT UI =================
st.set_page_config(page_title="My Annoying Shrink", layout="centered")

st.title("🧠 My Annoying Shrink")
st.write("Enter text and detect emotions")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

model = create_model()
model = load_model(model)

user_text = st.text_input("Enter your text")

if user_text:

    emotions, probs, best = predict_sample(user_text, model, tokenizer)

    st.subheader("💡 Expert Response")
    st.write(mapping_emotions[best][1])

    if st.button("Show emotions"):

        st.subheader("Detected emotions")

        for e, p in zip(emotions, probs):
            st.write(f"{mapping_emotions[e][0]} {e} → {p}")
