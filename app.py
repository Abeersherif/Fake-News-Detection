import os
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- PyTorch & BERT ----
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import classification_report

# Disable torch dynamo to avoid META device bug
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

###############################################################
# PAGE CONFIG
###############################################################
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection System (CNN / LSTM / InceptionResNet / BERT)")
st.caption("✅ Optimized Models Loaded (Mixed Precision / Balanced Training)")

st.write("🔍 Current working directory:", os.getcwd())
if os.path.exists("models"):
    st.write("📁 Files inside `models/`:", os.listdir("models"))
else:
    st.write("❌ models/ folder NOT FOUND!")

###############################################################
# TEXT CLEANING (same as training)
###############################################################
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


###############################################################
# LOAD CNN / LSTM / InceptionResNet (Keras Models) — ROBUST
###############################################################
def load_keras_model(
    model_path,
    tokenizer_path="models/tokenizer.pickle",
    model_kind: str | None = None,
):
    """
    1) Try to open different candidates for the model.
    2) Load tokenizer from the provided path.
    """
    candidates = []

    # 1) Direct path to model
    candidates.append(model_path)

    # 2) Append ".keras" extension to model name
    base, ext = os.path.splitext(model_path)
    alt_path = base + ".keras"
    if alt_path != model_path:
        candidates.append(alt_path)

    # 3) Search in the models/ folder for models
    models_dir = "models"
    if model_kind and os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            lower = fname.lower()
            if model_kind.lower() in lower and lower.endswith((".h5", ".keras")):
                full = os.path.join(models_dir, fname)
                candidates.append(full)

    # Remove duplicates and keep order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # If tokenizer is missing, skip loading
    if not os.path.exists(tokenizer_path):
        st.write(f"❌ Tokenizer not found at: {tokenizer_path}")
        return None, None

    # Try loading models
    last_error = None
    for path in unique_candidates:
        if not os.path.exists(path):
            continue
        try:
            st.write(f"🔎 Trying to load Keras model from: {path}")
            model = tf.keras.models.load_model(path)
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            st.write(f"✅ Loaded Keras model from: {path}")
            return model, tokenizer
        except Exception as e:
            st.write(f"⚠️ Failed loading {path}: {e}")
            last_error = e

    st.write("❌ Could not load any Keras model candidate.")
    if last_error:
        st.write(f"Last error: {last_error}")
    return None, None


###############################################################
# LOAD BERT MODEL (PyTorch)
###############################################################
def load_bert_model():
    model_dir = "models/welfake_bert_model_dir"
    tok_dir = "models/welfake_bert_tokenizer_dir"
    pt_path = "models/welfake_bert_model.pt"

    if not (os.path.exists(model_dir) and os.path.exists(tok_dir)):
        st.write("❌ BERT directories not found.")
        return None, None

    st.write("🔎 Loading BERT tokenizer from:", tok_dir)
    tokenizer = BertTokenizerFast.from_pretrained(tok_dir)

    st.write("🔎 Loading BERT base model from:", model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    )
    model.to("cpu")

    if os.path.exists(pt_path):
        st.write("🔎 Loading fine-tuned BERT weights from:", pt_path)
        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        st.write("⚠️ Fine-tuned weights not found, using base model only.")

    model.eval()
    st.write("✅ BERT model ready on CPU.")
    return model, tokenizer


###############################################################
# PREDICT WITH CNN / LSTM
###############################################################
def predict_keras(model, tokenizer, text, max_len=200) -> float:
    """
    Returns:
      prob_fake: Probability that the news is Fake
    """
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    
    
    # TF Models now trained on Target = Label (0=Real, 1=Fake)
    # Output is P(Fake) directly.
    prob_fake = model.predict(padded)[0][0]
    
    return float(prob_fake)


###############################################################
# PREDICT WITH BERT
###############################################################
def predict_bert(model, tokenizer, text, max_len=160) -> float:
    """
    Returns:
      prob_fake: Probability that the news is Fake (class 1)
    (softmax on logits: index 1 = Fake)
    """
    if model is None or tokenizer is None:
        return 0.5

    # BERT was trained on RAW text (see train_bert_welfake in train.py),
    # so we should NOT clean it (stripping punctuation etc.)
    # cleaned = clean_text(text)

    encoded = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    device = next(model.parameters()).device  # should be cpu
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    logits = outputs.logits 
    # BERT trained on Labels: 0=Real, 1=Fake.
    # So Logits Index 0 = Real, Index 1 = Fake.
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    prob_fake = float(probs[1])  # class 1 = Fake
    return prob_fake


###############################################################
# MODEL SELECTION
###############################################################
st.subheader("Choose model:")
model_choice = st.selectbox(
    "Select a model to test:",
    ["CNN", "LSTM", "InceptionResNet", "BERT", "🔮 Compare All Models"],
)

st.markdown("---")


###############################################################
# LOAD MODELS ACCORDING TO USER CHOICE
###############################################################
cnn_model = cnn_tok = None
lstm_model = lstm_tok = None
bert_model = bert_tok = None
inc_model = inc_tok = None

# Load CNN model if selected
if model_choice in ["CNN", "🔮 Compare All Models"]:
    cnn_model, cnn_tok = load_keras_model(
        "models/cnn_model.h5",
        tokenizer_path="models/tokenizer.pickle",
        model_kind="cnn",
    )

# Load LSTM model if selected
if model_choice in ["LSTM", "🔮 Compare All Models"]:
    lstm_model, lstm_tok = load_keras_model(
        "models/lstm_model.h5",
        tokenizer_path="models/tokenizer.pickle",
        model_kind="lstm",
    )

# Load BERT model if selected
if model_choice in ["BERT", "🔮 Compare All Models"]:
    bert_model, bert_tok = load_bert_model()

# Load InceptionResNet model if selected
if model_choice in ["InceptionResNet", "🔮 Compare All Models"]:
    inc_model, inc_tok = load_keras_model(
        "models/inception_resnet_model.h5",
        tokenizer_path="models/tokenizer.pickle",
        model_kind="inception_resnet",
    )

# Display model loading status
if model_choice == "CNN":
    if cnn_model is None:
        st.error("⚠️ CNN model could not be loaded. Check models folder & filenames.")
        st.stop()
    else:
        st.success("✅ CNN model loaded.")

elif model_choice == "LSTM":
    if lstm_model is None:
        st.error("⚠️ LSTM model could not be loaded. Check models folder & filenames.")
        st.stop()
    else:
        st.success("✅ LSTM model loaded.")

elif model_choice == "BERT":
    if bert_model is None:
        st.error("⚠️ BERT model could not be loaded. Check folders & weights.")
        st.stop()
    else:
        st.success("✅ BERT model loaded.")

elif model_choice == "InceptionResNet":
    if inc_model is None:
        st.error("⚠️ InceptionResNet model could not be loaded. Check models folder & filenames.")
        st.stop()
    else:
        st.success("✅ InceptionResNet model loaded.")

else:  # Compare all
    available = []
    if cnn_model is not None:
        available.append("CNN")
    if lstm_model is not None:
        available.append("LSTM")
    if bert_model is not None:
        available.append("BERT")
    if inc_model is not None:
        available.append("InceptionResNet")

    if not available:
        st.error("⚠️ No models could be loaded. Please check your models directory.")
        st.stop()
    else:
        st.success(f"✅ Loaded models: {', '.join(available)}")


###############################################################
# TEXT INPUT
###############################################################
user_text = st.text_area("Enter the article text:", height=200)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()

    results = {}

    with st.spinner("Analyzing..."):
        # CNN
        if model_choice in ["CNN", "🔮 Compare All Models"] and cnn_model is not None:
            p_fake = predict_keras(cnn_model, cnn_tok, user_text)
            results["CNN"] = p_fake

        # LSTM
        if model_choice in ["LSTM", "🔮 Compare All Models"] and lstm_model is not None:
            p_fake = predict_keras(lstm_model, lstm_tok, user_text)
            results["LSTM"] = p_fake

        # BERT
        if model_choice in ["BERT", "🔮 Compare All Models"] and bert_model is not None:
            p_fake = predict_bert(bert_model, bert_tok, user_text)
            results["BERT"] = p_fake

        # InceptionResNet
        if model_choice in ["InceptionResNet", "🔮 Compare All Models"] and inc_model is not None:
            p_fake = predict_keras(inc_model, inc_tok, user_text)
            results["InceptionResNet"] = p_fake

    st.markdown("## 🧪 Results")

    if not results:
        st.error("No model produced a prediction.")
    else:
        for model_name, prob_fake in results.items():
            # 0 = Real, 1 = Fake  →  prob_fake = P(Fake)
            prob_real = 1.0 - prob_fake
            is_fake = prob_fake > 0.5

            confidence = prob_fake if is_fake else prob_real
            label = "🔴 FAKE (1)" if is_fake else "🟢 REAL (0)"

            st.write(
                f"### **{model_name} → {label}** "
                f"(P(Fake) = {prob_fake:.2%}, P(Real) = {prob_real:.2%}, "
                f"Decision confidence: **{confidence:.2%}**) "
            )

        st.success("Done ✅")
