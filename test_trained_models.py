
import os
import tensorflow as tf
import torch
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizerFast, BertForSequenceClassification

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def test_models():
    print("Loading Models for Diagnostics...")
    
    # 1. Load Keras Logic
    # 0=Real, 1=Fake
    # Expected: Real < 0.5, Fake > 0.5
    
    samples = {
        "Real_Sample": "The city of Paris is the capital of France. The Eiffel Tower is a famous landmark located there.",
        "Fake_Sample": "BREAKING: Aliens have landed in New York City and are distributing free pizza to all residents!"
    }
    
    # Load Tokenizer
    if not os.path.exists("models/tokenizer.pickle"):
        print("No tokenizer found! Training probably failed.")
        return
        
    with open("models/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
        
    # Helper for Keras
    def predict_keras(name, model_path):
        if not os.path.exists(model_path):
             print(f"Warning: {name} model missing.")
             return
             
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"\n--- {name} Results ---")
            for label, text in samples.items():
                cleaned_text = clean_text(text)
                seq = tokenizer.texts_to_sequences([cleaned_text])
                pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                prob = model.predict(pad, verbose=0)[0][0]
                # If training was correct (0=Real, 1=Fake):
                # Real should be low (e.g. 0.1), Fake should be high (e.g. 0.9)
                print(f"  [{label}] Raw Output: {prob:.4f} => {'FAKE' if prob > 0.5 else 'REAL'}")
        except Exception as e:
            print(f"Error {name}: {e}")

    predict_keras("CNN", "models/cnn_model.h5")
    predict_keras("LSTM", "models/lstm_model.h5")
    predict_keras("Inception", "models/inception_resnet_model.h5")
    
    # 2. Load BERT Logic
    print("\n--- BERT Results ---")
    bert_dir = "models/welfake_bert_model_dir"
    bert_pt = "models/welfake_bert_model.pt"
    
    if os.path.exists(bert_dir):
        try:
            bert_tok = BertTokenizerFast.from_pretrained("models/welfake_bert_tokenizer_dir")
            bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
            if os.path.exists(bert_pt):
                 bert_model.load_state_dict(torch.load(bert_pt, map_location="cpu"))
            
            bert_model.eval()
            
            for label, text in samples.items():
                enc = bert_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=160)
                with torch.no_grad():
                    out = bert_model(**enc)
                    probs = torch.softmax(out.logits, dim=-1)[0] # [P(Real), P(Fake)]
                    prob_fake = float(probs[1])
                
                print(f"  [{label}] P(Fake): {prob_fake:.4f} => {'FAKE' if prob_fake > 0.5 else 'REAL'}")
                
        except Exception as e:
             print(f"BERT Error: {e}")
    else:
        print("BERT model folder missing.")

if __name__ == "__main__":
    test_models()
