import re
import contractions
import emoji


# Text preprocessing
def preprocess_corpus(x):
    
    x = re.sub(r'([a-zA-Z\[\]])([,;.!?])', r'\1 \2', x)
    x = re.sub(r'([,;.!?])([a-zA-Z\[\]])', r'\1 \2', x)

    x = emoji.demojize(x)
    x = contractions.fix(x)
    x = x.lower()

    x = re.sub(r"\b(idk)\b", "i do not know", x)
    x = re.sub(r"\b(tho)\b", "though", x)
    x = re.sub(r"\b(lmao)\b", "laughing my ass off", x)

    x = re.sub(r"[^A-Za-z!?_]+"," ", x)
    x = re.sub(r" +"," ", x)
    x = x.strip()

    return x


# ✅ FIXED TOKENIZER
def tokenize(tokenizer, text_samples):
    sample_token = tokenizer(
        text=text_samples,
        add_special_tokens=True,
        max_length=48,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=True,
        return_attention_mask=True
    )

    return {
        'input_ids': sample_token['input_ids'],
        'attention_mask': sample_token['attention_mask'],
        'token_type_ids': sample_token['token_type_ids']  # ✅ FIXED
    }
