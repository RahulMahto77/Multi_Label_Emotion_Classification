import re
import contractions
import emoji


def preprocess_corpus(x):

    x = re.sub(r'([a-zA-Z\[\]])([,;.!?])', r'\1 \2', x)
    x = re.sub(r'([,;.!?])([a-zA-Z\[\]])', r'\1 \2', x)

    x = emoji.demojize(x)
    x = contractions.fix(x)
    x = x.lower()

    x = re.sub(r"lmao", "laughing my ass off", x)
    x = re.sub(r"\b(tho)\b", "though", x)
    x = re.sub(r"\b(ikr)\b", "i know right", x)
    x = re.sub(r"\b(ya|u)\b", "you", x)

    x = re.sub(r"\b(idk)\b", "i do not know", x)
    x = re.sub(r"\b(bruh)\b", "bro", x)

    x = re.sub(r"<3", " love ", x)
    x = re.sub(r":\)", " smiling_face ", x)
    x = re.sub(r":\(", " frowning_face ", x)

    x = re.sub(r"[^A-Za-z!?_]+", " ", x)
    x = re.sub(r" +", " ", x).strip()

    return x


def tokenize(tokenizer, text_samples):
    sample_token = tokenizer(
        text=text_samples,
        add_special_tokens=True,
        max_length=48,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_attention_mask=True
    )

    return {
        "input_ids": sample_token["input_ids"],
        "attention_mask": sample_token["attention_mask"]
    }
