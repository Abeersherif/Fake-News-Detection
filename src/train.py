import os
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_manager import DataManager
from src.utils import plot_confusion_matrix_heatmap


def train_and_evaluate(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    epochs: int = 5,
    batch_size: int = 32,
) -> Tuple[tf.keras.callbacks.History, float, float]:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", f"{model_name}_best.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"[{model_name}] Starting training for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    train_acc_list = history.history.get("accuracy", [])
    if len(train_acc_list) > 0:
        train_acc = float(train_acc_list[-1])
    else:
        train_acc = float("nan")

    val_acc_list = history.history.get("val_accuracy", [])
    if len(val_acc_list) > 0:
        final_val_acc = float(val_acc_list[-1])
    else:
        final_val_acc = float("nan")

    print(f"[{model_name}] Final Training Accuracy (last epoch): {train_acc:.4f}")
    if not (final_val_acc != final_val_acc):
        print(f"[{model_name}] Final Validation Accuracy (last epoch): {final_val_acc:.4f}")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return history, train_acc, test_acc


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 160):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_bert_welfake(
    max_samples: int | None = None,
    model_name: str = "bert-base-uncased",
    max_len: int = 160,
    batch_size: int = 8,
    epochs: int = 2,
    pretrained: bool = False,
):
    print("Loading dataset with DataManager...")
    dm = DataManager()
    df = dm.download_data(sample_frac=0.5)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'text' and 'label' in WELFake dataset.")

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using a subset of {max_samples} samples for BERT training.")
    else:
        print(f"Using ALL {len(df)} samples for BERT training.")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
        )
    print(
        f"BERT split: {len(X_train)} samples for training (~80%) and "
        f"{len(X_val)} samples for validation/testing (~20%)."
    )

    print("Loading BERT tokenizer + model...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    if pretrained:
        for param in model.embeddings.parameters():
            param.requires_grad = False
        print("[BERT] Embedding layer frozen for pretrained model.")

    device = "cpu"
    print("Using device: CPU")
    model.to(device)

    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_dataset = BERTDataset(X_val, y_val, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    last_train_acc = 0.0
    last_val_acc = 0.0
    last_val_labels: List[int] = []
    last_val_preds: List[int] = []

    print("Starting BERT training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        train_correct = 0
        train_total = 0

        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

            if step % 50 == 0 or step == num_batches:
                avg_loss = total_loss / step
                print(f"  Step {step}/{num_batches} - Avg Loss: {avg_loss:.4f}")

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        last_train_acc = train_acc

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

                all_val_labels.extend(labels_batch.cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        last_val_acc = val_acc

        last_val_labels = all_val_labels
        last_val_preds = all_val_preds

        print(
            f"Epoch {epoch + 1} - "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        print("\n[BERT] Validation Classification Report:")
        print(
            classification_report(
                all_val_labels,
                all_val_preds,
                target_names=["Real (0)", "Fake (1)"],
                digits=4,
            )
        )

    print("\nBERT training complete!")
    print(
        f"[BERT_Model] Final Training Accuracy: {last_train_acc:.4f} | "
        f"Final Validation/Test Accuracy: {last_val_acc:.4f}"
    )

    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix_heatmap(
        y_true=last_val_labels,
        y_pred=last_val_preds,
        model_name="BERT_Model",
        save_dir="outputs",
    )
    print("[BERT] Final confusion matrix saved to outputs/BERT_Model_cm.png")

    os.makedirs("models", exist_ok=True)

    model_dir = "models/welfake_bert_model_dir"
    tokenizer_dir = "models/welfake_bert_tokenizer_dir"

    print("Saving BERT model + tokenizer...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    torch.save(model.state_dict(), "models/welfake_bert_model.pt")

    print("Saved:")
    print(f" - model weights: models/welfake_bert_model.pt")
    print(f" - model folder: {model_dir}")
    print(f" - tokenizer folder: {tokenizer_dir}")

    return (
        model,
        tokenizer,
        last_train_acc,
        last_val_acc,
        len(X_train),
        len(X_val),
        last_val_labels,
        last_val_preds,
    )
