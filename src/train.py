import os
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision

# Enable Mixed Precision ONLY if GPU is present (Slow on CPU!)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"✅ GPU Detected: Enabled Mixed Precision (Turbo Mode).")
    except Exception as e:
        print(f"⚠️ GPU Mixed Precision Failed: {e}")
else:
    print("ℹ️ CPU Detected: Mixed Precision DISABLED (Avoiding slowdown).")

# ---- BERT (PyTorch) ----
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_manager import DataManager
from src.utils import plot_confusion_matrix_heatmap  # 🔹 For confusion matrix of BERT


# ============================================================
# 1) CNN / LSTM / Inception-Residual Training (TensorFlow / Keras)
# ============================================================
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
    """
    Compile + train model, then evaluate on test set.

    Returns:
        history   : Keras History object
        train_acc : final training accuracy (last epoch)
        test_acc  : accuracy on the held-out test set
    """

    # 1) Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 2) Callbacks (early stopping + best checkpoint per model)
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

    # 3) Final training accuracy from history
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
    if not (final_val_acc != final_val_acc):  # NaN check
        print(f"[{model_name}] Final Validation Accuracy (last epoch): {final_val_acc:.4f}")

    # 4) Evaluate on the held-out test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return history, train_acc, test_acc


# ============================================================
# 2) BERT Dataset (PyTorch)
# ============================================================
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 160):
        """
        texts  : list[str]
        labels : list[int]  (0 = Fake, 1 = Real)
        """
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


# ============================================================
# 3) BERT Training Pipeline (PyTorch) + Confusion Matrix (LAST epoch only)
# ============================================================
def train_bert_welfake(
    max_samples: int | None = None,
    model_name: str = "bert-base-uncased",
    max_len: int = 160,
    batch_size: int = 8,
    epochs: int = 2,
    pretrained: bool = False,  # Freeze early layers if pretrained
):
    """
    Train a BERT model on the WELFake dataset using DataManager.

    LABELS:
      - 0 = Fake
      - 1 = Real

    Uses an 80/20 train/validation split (validation ≈ testing set).

    If max_samples is None -> use ALL samples.
    If max_samples > 0    -> sample that many rows for faster debugging.

    Saves:
      - models/welfake_bert_model_dir
      - models/welfake_bert_tokenizer_dir
      - models/welfake_bert_model.pt
      - outputs/BERT_Model_cm.png (Confusion Matrix for LAST epoch only)

    Returns:
      model, tokenizer,
      final_train_acc, final_val_acc,
      train_size, val_size,
      last_val_labels (list[int]), last_val_preds (list[int])
    """

    print("Loading dataset with DataManager...")
    dm = DataManager()
    df = dm.download_data()

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'text' and 'label' in WELFake dataset.")

    # If max_samples is None → use all data
    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using a subset of {max_samples} samples for BERT training.")
    else:
        print(f"Using ALL {len(df)} samples for BERT training.")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()  # 0 = Fake, 1 = Real

    # 80% train / 20% val split
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
        num_labels=2,  # 0 = Fake, 1 = Real
    )

    if pretrained:
        # Freeze early layers (Embedding, etc.)
        for param in model.embeddings.parameters():
            param.requires_grad = False
        print("[BERT] Embedding layer frozen for pretrained model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # DataLoaders
    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_dataset = BERTDataset(X_val, y_val, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer & Scaler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()

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

            # Automatic Mixed Precision
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )
                loss = outputs.loss
                logits = outputs.logits

            # Scale loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Training accuracy tracking
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

            if step % 50 == 0 or step == num_batches:
                avg_loss = total_loss / step
                print(f"  Step {step}/{num_batches} - Avg Loss: {avg_loss:.4f}")

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        last_train_acc = train_acc

        # ============================
        # Validation
        # ============================
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

        # Save last epoch labels/preds for metrics & confusion matrix
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
                target_names=["Fake (0)", "Real (1)"],
                digits=4,
            )
        )

    print("\nBERT training complete!")
    print(
        f"[BERT_Model] Final Training Accuracy: {last_train_acc:.4f} | "
        f"Final Validation/Test Accuracy: {last_val_acc:.4f}"
    )

    # ============================
    # Confusion Matrix (LAST epoch only)
    # ============================
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix_heatmap(
        y_true=last_val_labels,
        y_pred=last_val_preds,
        model_name="BERT_Model",
        save_dir="outputs",
    )
    print("[BERT] Final confusion matrix saved to outputs/BERT_Model_cm.png")

    # ============================
    # 6) Save model + tokenizer
    # ============================
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
