import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

import myfuncs


# ---------------- CONFIG ---------------- #
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
    "annoyance":["😤","It’s annoying, I know."],
    "approval":["👍","Good choice!"],
    "caring":["🥰","That’s caring."],
    "confusion":["🤔","You seem confused."],
    "curiosity":["🧐","Curiosity is good!"],
    "desire":["😋","You really want it."],
    "disappointment":["😞","That’s disappointing."],
    "disapproval":["🙄","Not approved."],
    "disgust":["🤢","That’s disgusting."],
    "embarrassment":["😳","Awkward moment."],
    "excitement":["😆","You are excited!"],
    "fear":["😱","Something is scary."],
    "gratitude":["🙏","Be thankful."],
    "grief":["🥀","That’s sad."],
    "joy":["😄","You are happy!"],
    "love":["❤️","Love is in the air."],
    "nervousness":["😬","Feeling nervous."],
    "optimism":["💪","Stay positive."],
    "pride":["😎","Be proud!"],
    "realization":["💡","You understood something."],
    "relief":["😌","Feeling relieved."],
    "remorse":["😔","You regret it."],
    "sadness":["😭","That’s sad."],
    "surprise":["😮","Surprising!"],
    "neutral":["😐","Nothing special detected."]
}


# ---------------- MODEL ---------------- #
config = BertConfig.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def create_model(nb_labels, max_length=48):

    transformer = TFBertModel.from_pretrained("bert-base-uncased", config=config)

    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")

    bert_output = transformer(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[1]   # pooled output

    x = Dropout(0.3)(bert_output)

    output = Dense(
        nb_labels,
        activation="sigmoid",
        kernel_initializer=TruncatedNormal(stddev=0.02)
    )(x)

    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=output
    )

    return model


# ❌ FIX: REMOVE STREAMLIT CACHE FOR MODEL
def load_model(model, weights_path):
    model.load_weights(weights_path)
    return model


# ---------------- PREDICT ---------------- #
def predict_sample(text, model, threshold=0.87):

    text = myfuncs.preprocess_corpus(text)
    tokens = myfuncs.tokenize(tokenizer, text)

    probs = model.predict(tokens, verbose=0)[0]

    labels = [GE_taxonomy[i] for i, p in enumerate(probs) if p > threshold]
    scores = [p for p in probs if p > threshold]

    if len(labels) == 0:
        return ["neutral"], [1.0], "neutral"

    best_idx = np.argmax(probs)
    best_label = GE_taxonomy[best_idx]

    return labels, scores, best_label


# ---------------- STREAMLIT APP ---------------- #
st.set_page_config(page_title="Emotion App", layout="centered")

st.title("🧠 My Annoying Shrink")

text = st.text_input("How are you feeling today?")

@st.cache_resource
def load_full_model():
    model = create_model(27)
    model = load_model(model, "bert-weights.hdf5")
    return model

model = load_full_model()


if text:

    emotions, probs, best = predict_sample(text, model)

    st.subheader("Detected Emotion")
    st.write(mapping_emotions[best][0], best)

    st.subheader("Response")
    st.write(mapping_emotions[best][1])

    st.subheader("All emotions")
    for e, p in zip(emotions, probs):
        st.write(f"{e} → {p:.2f}")
