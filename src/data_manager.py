import os
import re
import random
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, max_words: int = 10000, max_len: int = 200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    # --------------------------------------------------
    # 1) Load / download data
    # --------------------------------------------------
    def download_data(
        self,
        local_path: str = "WELFake_Dataset.csv",
        sample_frac: float = 0.5,
    ) -> pd.DataFrame:       
        if os.path.exists(local_path):
            print(f" Using local dataset: {local_path}")
            df = pd.read_csv(local_path)
            print(f"⚠️ Data sampling: Using {sample_frac*100}% data...")
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            return self._normalize_columns(df)
        raise FileNotFoundError(
            f"❌ Could not find dataset at '{local_path}'. "
            "Please ensure 'WELFake_Dataset.csv' is in the project root."
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop unnamed index column if it exists
        for col in df.columns:
            if "unnamed" in col.lower():
                df = df.drop(columns=[col])

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Check required columns exist
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        # Ensure label is numeric
        df["label"] = df["label"].astype(int)
        return df

    # --------------------------------------------------
    # 2) Cleaning / augmentation
    # --------------------------------------------------
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    def augment_text(self, text: str) -> str:
        words = text.split()
        if len(words) > 5:
            return " ".join([w for w in words if random.random() > 0.1])
        return text

    # --------------------------------------------------
    # 3) Prepare train/test data
    # --------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, augment: bool = True):
        # Ensure required columns exist
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError("Dataframe must contain 'text' and 'label' columns.")

        # Drop rows with missing text/label
        df = df.dropna(subset=["text", "label"])

        # Clean text
        df["clean_text"] = df["text"].apply(self.clean_text)

        # Augmentation
        if augment:
            augmented_texts = df["clean_text"].apply(self.augment_text)
            df_aug = pd.DataFrame({"clean_text": augmented_texts, "label": df["label"]})
            df = pd.concat([df, df_aug], ignore_index=True)

        # Label encoding (verified by check_labels.py):
        # 0 = Real news
        # 1 = Fake news
        df["label"] = df["label"].astype(int)
        df["target"] = df["label"]  # 0=Real, 1=Fake

        # Tokenize + pad
        self.tokenizer.fit_on_texts(df["clean_text"])
        sequences = self.tokenizer.texts_to_sequences(df["clean_text"])
        padded = pad_sequences(
            sequences, maxlen=self.max_len, padding="post", truncating="post"
        )

        X = padded
        y = df["target"].values

        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
