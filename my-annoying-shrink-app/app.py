import numpy as np
from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal
import streamlit as st
import myfuncs


# ---------------- LABELS ----------------
GE_taxonomy = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
                "joy", "love", "nervousness", "optimism", "pride", "realization", 
                "relief", "remorse", "sadness", "surprise", "neutral"]


# ---------------- EMOTION MAPPING ----------------
mapping_emotions = {
  "admiration":["😍","You admire something."],
  "amusement":["🥳","You seem amused!"], 
  "anger":["😡","Take a breath, it's okay."], 
  "annoyance":["😤","That sounds annoying."], 
  "approval":["👍","Looks like approval."], 
  "caring":["😘","You care deeply."], 
  "confusion":["🤨","Seems confusing right?"], 
  "curiosity":["🤔","You're curious!"], 
  "desire":["🤤","Strong desire detected."], 
  "disappointment":["😔","That's disappointing."], 
  "disapproval":["🙄","Not impressed, huh?"], 
  "disgust":["🤮","That's unpleasant."], 
  "embarrassment":["😳","That’s awkward."], 
  "excitement":["😆","You're excited!"], 
  "fear":["😱","Stay calm."], 
  "gratitude":["🙏","Grateful vibes."], 
  "grief":["🥀","Sorry for your loss."], 
  "joy":["🤗","That's joyful!"],
  "love":["😍","Love is in the air."], 
  "nervousness":["😬","Relax, you got this."], 
  "optimism":["💪","Stay positive!"], 
  "pride":["😎","Be proud!"], 
  "realization":["🏆","Now you see it."], 
  "relief":["😌","That’s a relief."], 
  "remorse":["😞","Regret happens."], 
  "sadness":["😭","Hope things get better."], 
  "surprise":["😮","Unexpected!"], 
  "neutral":["😐","Hmm... neutral."]
}


# ---------------- PREDICTION ----------------
def predict_sample(text_sample, model, tokenizer, threshold=0.87):

    text_sample = myfuncs.preprocess_corpus(text_sample)
    sample = myfuncs.tokenize(tokenizer, text_sample)

    sample_probas = model.predict(sample)
    sample_probas = sample_probas.ravel().tolist()

    sample_labels = [1 if (p > threshold) else 0 for p in sample_probas]
    best_idx = np.argmax(sample_probas)
    best_label = GE_taxonomy[best_idx]

    sample_labels = [GE_taxonomy[i] for i in range(len(sample_labels)) if sample_labels[i]==1]
    sample_probas = [p for p in sample_probas if p > threshold]

    if len(sample_labels) == 0:
        sample_labels = ["neutral"]
        sample_probas = [0]
        best_label = "neutral"

    return sample_labels, sample_probas, best_label


# ---------------- MODEL ----------------
config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


@st.cache_resource
def create_model(nb_labels, max_length=48):

    bert = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(max_length,), name='token_type_ids', dtype='int32')

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

    outputs = bert(inputs)
    pooled_output = outputs.pooler_output

    x = Dropout(0.3)(pooled_output)

    outputs = Dense(
        units=nb_labels,
        activation="sigmoid",
        kernel_initializer=TruncatedNormal(stddev=config.initializer_range)
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


@st.cache_resource
def load_model_weights(model):
    model.load_weights("bert-weights.hdf5")  # ⚠️ keep file in same folder
    return model


model = create_model(len(GE_taxonomy))
model = load_model_weights(model)


# ---------------- UI ----------------
st.title("🧠 Emotion Detection App")

text = st.text_input("Enter your text")

if text:
    emotions, probs, best = predict_sample(text, model, tokenizer)

    emoji_icon, message = mapping_emotions[best]

    st.markdown(f"## {emoji_icon} {best}")
    st.write(message)

    st.subheader("All detected emotions:")
    for e, p in zip(emotions, probs):
        st.write(f"{mapping_emotions[e][0]} {e} ({p:.2f})")
