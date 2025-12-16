import os
import re
import random
import requests
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
        save_path: str = "data/welfake_dataset.csv",
    ) -> pd.DataFrame:
        """
        1) If local_path (WELFake_Dataset.csv) exists in project root → use it.
        2) Else if save_path (data/welfake_dataset.csv) exists → use it.
        3) Else try to download from Zenodo.
        4) If download fails → generate small synthetic dataset.
        """
        # 1) Local file in project root
        if os.path.exists(local_path):
            print(f" Using local dataset: {local_path}")
            df = pd.read_csv(local_path)
            print("⚠️ CPU BALANCED MODE: Using 20% data...")
            df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
            return self._normalize_columns(df)

        # 2) File in data/ folder
        if os.path.exists(save_path):
            print(f" Using dataset from: {save_path}")
            df = pd.read_csv(save_path)
            print("⚠️ CPU BALANCED MODE: Using 20% data...")
            df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
            return self._normalize_columns(df)

        # 3) Try to download
        url = "https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            print(f" Attempting to download data from {url} ...")
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f" Download successful. Saved to {save_path}")
            df = pd.read_csv(save_path)
            # CPU BALANCED MODE: Downsample to 20% (~14k samples)
            # 10% was too weak (0.94 acc). 20% w/o augmentation is a good middle ground.
            print("⚠️ CPU BALANCED MODE: Using 20% data (~14k samples) for better accuracy...")
            df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
            return self._normalize_columns(df)
        except Exception as e:
            print(f" Download failed: {e}. Generating synthetic data...")
            df = self.generate_synthetic_data(save_path)
            return self._normalize_columns(df)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure we always have: 'title', 'text', 'label'
        and drop the useless index column if present.
        """
        # Drop unnamed index column if it exists
        for col in df.columns:
            if "unnamed" in col.lower():
                df = df.drop(columns=[col])

        # Unify possible column casings
        col_map = {c.lower(): c for c in df.columns}
        # We expect these logical columns
        needed = ["title", "text", "label"]

        for name in needed:
            if name not in col_map:
                # If completely missing, create empty title / raise for others
                if name == "title":
                    df["title"] = ""
                else:
                    raise ValueError(f"Required column '{name}' not found in dataset.")
        # Rename to lowercase canonical form
        df = df.rename(
            columns={
                col_map.get("title", "title"): "title",
                col_map.get("text", "text"): "text",
                col_map.get("label", "label"): "label",
            }
        )

        # Ensure label numeric
        df["label"] = df["label"].astype(int)
        return df

    # --------------------------------------------------
    # 2) Fallback synthetic dataset
    # --------------------------------------------------
    def generate_synthetic_data(self, save_path: str) -> pd.DataFrame:
        """Generates a small synthetic dataset for testing."""
        print(" Creating synthetic dataset for test runs...")
        data = {
            "title": [f"News Title {i}" for i in range(100)],
            "text": [
                f"This is some sample news text content number {i} with enough words to test the model."
                for i in range(100)
            ],
            # mimic WELFake: 0=fake, 1=real
            "label": [0 if i % 2 == 0 else 1 for i in range(100)],
        }
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f" Synthetic dataset saved to {save_path}")
        return df

    # --------------------------------------------------
    # 3) Cleaning / augmentation
    # --------------------------------------------------
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    def augment_text(self, text: str) -> str:
        """Randomly deletes words for augmentation."""
        words = text.split()
        if len(words) > 5:
            return " ".join([w for w in words if random.random() > 0.1])
        return text

    # --------------------------------------------------
    # 4) Prepare train/test data
    # --------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, augment: bool = False):
        """
        - Clean text
        - Optional augmentation
        - Map labels if needed
        - Tokenize + pad
        - Split train/test
        """
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

        # WELFake: 0=fake, 1=real
        # We standardise to: 0=Real, 1=Fake
        # (Assuming original WELFake 0=Fake, 1=Real was actually 0=Real, 1=Fake based on my findings)
        # Based on my check_truth.py findings: 0=Real, 1=Fake.
        # So we just carry it over directly.
        df["label"] = df["label"].astype(int)
        df["target"] = df["label"]  # 0=Real, 1=Fake (Consistent!)

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
