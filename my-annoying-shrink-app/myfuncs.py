import re
import contractions
import emoji

def preprocess_corpus(x):
    x = emoji.demojize(x)
    x = contractions.fix(x)
    x = x.lower()
    x = re.sub(r"[^A-Za-z!?_]+"," ", x)
    x = re.sub(r" +"," ", x)
    return x.strip()

def tokenize(tokenizer, text):
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=48,
        return_tensors='tf'
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "token_type_ids": tokens["token_type_ids"]
    }
