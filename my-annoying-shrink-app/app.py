import os
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

import myfuncs


# ---------------- SETTINGS ---------------- #
st.set_page_config(page_title="My Annoying Shrink", layout="centered")

GE_taxonomy = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]


mapping_emotions = {
    "admiration":["😍","You admire something deeply."],
    "amusement":["🤪","That’s funny!"],
    "anger":["😡","Try to calm down."],
    "annoyance":["😤","It’s annoying."],
    "approval":["👍","Approved!"],
    "caring":["🥰","You care."],
    "confusion":["🤔","You are confused."],
    "curiosity":["🧐","Curious mind!"],
    "desire":["😋","You want it."],
    "disappointment":["😞","That’s sad."],
    "disapproval":["🙄","Not approved."],
    "disgust":["🤢","Disgusted."],
    "embarrassment":["😳","Embarrassed."],
    "excitement":["😆","Very excited!"],
    "fear":["😱","Fear detected."],
    "gratitude":["🙏","Be thankful."],
    "grief":["🥀","Deep sadness."],
    "joy":["😄","Happy!"],
    "love":["❤️","Love detected."],
    "nervousness":["😬","Nervous."],
    "optimism":["💪","Stay positive."],
    "pride":["😎","Be proud."],
    "realization":["💡","Realization."],
    "relief":["😌","Relieved."],
    "remorse":["😔","Regret."],
    "sadness":["😭","Sad."],
    "surprise":["😮","Surprised!"],
    "neutral":["😐","Neutral state."]
}


# ---------------- MODEL SETUP ---------------- #
config = BertConfig.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def create_model(nb_labels, max_length=48):

    bert = TFBertModel.from_pretrained("bert-base-uncased", config=config)

    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")

    bert_output = bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[1]  # pooled output

    x = Dropout(0.3)(bert_output)

    output = Dense(
        nb_labels,
        activation="sigmoid",
        kernel_initializer=TruncatedNormal(stddev=0.02)
    )(x)

    return Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=output
    )


# ---------------- LOAD WEIGHTS SAFELY ---------------- #
def load_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Weights file not found: {path}")
    model.load_weights(path)
    return model


# ---------------- PREDICTION ---------------- #
def predict_sample(text, model, threshold=0.87):

    text = myfuncs.preprocess_corpus(text)
    tokens = myfuncs.tokenize(tokenizer, text)

    probs = model.predict(tokens, verbose=0)[0]

    labels = [GE_taxonomy[i] for i, p in enumerate(probs) if p > threshold]
    scores = [p for p in probs if p > threshold]

    if len(labels) == 0:
        return ["neutral"], [1.0], "neutral"

    best_idx = int(np.argmax(probs))
    best_label = GE_taxonomy[best_idx]

    return labels, scores, best_label


# ---------------- STREAMLIT UI ---------------- #
st.title("🧠 My Annoying Shrink")

text = st.text_input("How are you feeling today?")


# ---------------- LOAD MODEL (CACHE SAFE) ---------------- #
@st.cache_resource
def load_full_model():

    model = create_model(27)

    weights_path = os.path.join(
        os.path.dirname(__file__),
        "bert-weights.hdf5"
    )

    model = load_model(model, weights_path)
    return model


model = load_full_model()


# ---------------- RUN ---------------- #
if text:

    emotions, probs, best = predict_sample(text, model)

    st.subheader("Detected Emotion")
    st.write(mapping_emotions[best][0], best)

    st.subheader("Response")
    st.write(mapping_emotions[best][1])

    st.subheader("All detected emotions")

    for e, p in zip(emotions, probs):
        st.write(f"{e} → {p:.2f}")
