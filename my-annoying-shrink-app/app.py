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
    "admiration": ["🤩", "You always admire what you really don't understand."],
    "amusement": ["🥳", "The unfortunate who has to travel for amusement lacks capacity for amusement."],
    "anger": ["😡", "Bitterness is like fire. It burns it all clean."],
    "annoyance": ["😤", "It could be worse."],
    "approval": ["👍", "A man cannot be comfortable without his own approval."],
    "caring": ["😘", "The road to hell is paved with good intentions."],
    "confusion": ["🤨", "It's ok to be confused, but know why..."],
    "curiosity": ["🤔", "The cure for boredom is curiosity."],
    "desire": ["🤤", "From desire comes hate."],
    "disappointment": ["😔", "Grass is always greener."],
    "disapproval": ["🙄", "Don't judge a book by its cover."],
    "disgust": ["🤮", "Some people are beautifully wrapped boxes of sh*t."],
    "embarrassment": ["😳", "Shame is a soul eating emotion."],
    "excitement": ["😆", "Excitement always leads to tears."],
    "fear": ["😱", "Limits are often illusions."],
    "gratitude": ["🙏", "Gratitude is powerful."],
    "grief": ["🥀", "All good things end."],
    "joy": ["😊", "Everything is fine 🙂"],
    "love": ["😍", "Love turns into everything."],
    "nervousness": ["😬", "Anxiety mode on."],
    "optimism": ["💪", "Hope for the best."],
    "pride": ["😎", "Among the blind, one-eyed is king."],
    "realization": ["🏁", "Don’t blow your own trumpet."],
    "relief": ["😮‍💨", "Phewwww"],
    "remorse": ["😞", "You can’t undo mistakes."],
    "sadness": ["😭", "It’s okay to feel sad."],
    "surprise": ["😮", "Surprise!"],
    "neutral": ["😐", "Neutral state"]
}


# ================= PREDICTION =================
def predict_sample(text, model, tokenizer, threshold=0.85):

    text = myfuncs.preprocess_corpus(text)
    sample = myfuncs.tokenize(tokenizer, text)

    probs = model.predict(sample, verbose=0)[0]
    probs = probs.tolist()

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

    bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)
    pooled = bert_output.pooler_output

    x = Dropout(0.3)(pooled)
    output = Dense(27, activation="sigmoid",
                   kernel_initializer=TruncatedNormal(stddev=config.initializer_range))(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    return model


@st.cache_resource
def load_model(model):
    model.load_weights("bert-weights.hdf5")
    return model


# ================= STREAMLIT UI =================
st.set_page_config(page_title="My Annoying Shrink", layout="centered")

st.title("🧠 My Annoying Shrink")
st.write("Type your feelings and I will analyze your emotions 😄")


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

model = create_model()
model = load_model(model)

user_text = st.text_input("Enter your text here:")

if user_text:

    emotions, probs, best = predict_sample(user_text, model, tokenizer)

    st.subheader("💡 Expert Response")
    st.write(mapping_emotions[best][1])

    if st.button("Show emotions"):
        st.subheader("Detected emotions")

        for e, p in zip(emotions, probs):
            st.write(f"{mapping_emotions[e][0]} {e} → {p}")
