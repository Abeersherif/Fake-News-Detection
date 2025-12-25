import os
# Suppress TensorFlow INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from src.data_manager import DataManager
from src.models import build_model
from src.train import train_and_evaluate, train_bert_welfake
from src.utils import (
    save_artifacts,
    plot_confusion_matrix_heatmap,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN + LSTM + InceptionResNet + BERT on WELFake"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (Default: 5 for better accuracy)")
    parser.add_argument("--bert_epochs", type=int, default=1, help="Epochs for BERT (CPU Fast: 1)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CNN/LSTM/InceptionResNet")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ============================================================
    # 1) Load & prepare data
    # ============================================================
    print("\n==============================")
    print(" STEP 1 — Load & prepare data ")
    print("==============================\n")

    dm = DataManager()
    
    # -------------------------------------------------------------
    # 🚀 HIGH ACCURACY MODE (Hardcoded)
    # Using 100% Data + Augmentation + 5 Epochs
    # -------------------------------------------------------------
    print("🚀 HIGH ACCURACY MODE: Using 50% data + Augmentation + 5 Epochs.")
    print("✨ AUGMENTATION ENABLED: Dataset size will be doubled (effective ~80-100% of original).")

    sample_frac = 0.5  # Upgrade to 50% data as requested
    augment_data = True # Keep augmentation enabled

    df = dm.download_data(sample_frac=sample_frac)
    
    X_train, X_test, y_train, y_test = dm.prepare_data(df, augment=augment_data)

    vocab_size = len(dm.tokenizer.word_index) + 1
    print(f"Vocabulary Size : {vocab_size}")
    print(f"Train Samples   : {len(X_train)}")
    print(f"Test Samples    : {len(X_test)}")
    print(f"Unique labels   : {np.unique(y_train)}  (0 = Real, 1 = Fake)")

    # ---------- Save the 20% testing data to CSV ----------
    print("\nSaving the 20% testing data (text + label)...")
    test_texts = dm.tokenizer.sequences_to_texts(X_test)
    testing_df = pd.DataFrame({"text": test_texts, "label": y_test})
    testing_csv_path = os.path.join("outputs", "testing_data.csv")
    testing_df.to_csv(testing_csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Testing data saved to: {testing_csv_path}")
    print("\nSample of testing data (first 10 rows):")
    print(testing_df.head(10))

    # For global metrics table
    metrics_rows = []

    def add_metrics_rows(model_name: str, y_true, y_pred, test_acc: float):
        """
        Add per-class metrics (precision, recall, F1, support, accuracy)
        for Fake (0) and Real (1) to metrics_rows.
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        accuracy = report["accuracy"]
        for label_id, label_name in [("0", "Fake"), ("1", "Real")]:
            if label_id not in report:
                continue
            stats = report[label_id]
            metrics_rows.append(
                {
                    "model": model_name,
                    "class": label_name,
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1-score"],
                    "support": stats["support"],
                    "accuracy": accuracy,
                }
            )

    # ============================================================
    # 2) Train CNN
    # ============================================================
    print("\n====================")
    print(" TRAINING: CNN Model")
    print("====================\n")

    cnn_model = build_model("CNN", vocab_size)
    cnn_history, cnn_train_acc, cnn_test_acc = train_and_evaluate(
        cnn_model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="CNN_Model",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Confusion Matrix for CNN
    print("\n[ CNN ] Computing confusion matrix...")
    cnn_probs = cnn_model.predict(X_test, batch_size=args.batch_size).ravel()  # P(Real = 1)
    cnn_preds = (cnn_probs >= 0.5).astype(int)  # 0 = Fake, 1 = Real
    plot_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=cnn_preds,
        model_name="CNN_Model",
        save_dir="outputs",
    )
    print("[ CNN ] Confusion matrix saved to outputs/CNN_Model_cm.png")

    # Classification report for CNN
    print("\n[ CNN ] Classification report (0 = Real, 1 = Fake):")
    print(
        classification_report(
            y_test,
            cnn_preds,
            target_names=["Real (0)", "Fake (1)"],
            digits=4,
        )
    )

    # Save CNN model & tokenizer explicitly
    print("[ CNN ] Saving full CNN model + tokenizer...")
    save_artifacts(cnn_model, dm.tokenizer, model_name="cnn_model")

    add_metrics_rows("CNN", y_test, cnn_preds, cnn_test_acc)

    # ============================================================
    # 3) Train LSTM
    # ============================================================
    print("\n=====================")
    print(" TRAINING: LSTM Model")
    print("=====================\n")

    lstm_model = build_model("LSTM", vocab_size)
    lstm_history, lstm_train_acc, lstm_test_acc = train_and_evaluate(
        lstm_model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="LSTM_Model",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Confusion Matrix for LSTM
    print("\n[ LSTM ] Computing confusion matrix...")
    lstm_probs = lstm_model.predict(X_test, batch_size=args.batch_size).ravel()
    lstm_preds = (lstm_probs >= 0.5).astype(int)
    plot_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=lstm_preds,
        model_name="LSTM_Model",
        save_dir="outputs",
    )
    print("[ LSTM ] Confusion matrix saved to outputs/LSTM_Model_cm.png")

    print("\n[ LSTM ] Classification report (0 = Real, 1 = Fake):")
    print(
        classification_report(
            y_test,
            lstm_preds,
            target_names=["Fake (0)", "Real (1)"],
            digits=4,
        )
    )

    print("[ LSTM ] Saving full LSTM model + tokenizer...")
    save_artifacts(lstm_model, dm.tokenizer, model_name="lstm_model")

    add_metrics_rows("LSTM", y_test, lstm_preds, lstm_test_acc)

    # ============================================================
    # 4) Train Inception + Residual (InceptionResNet)
    # ============================================================
    print("\n====================================")
    print(" TRAINING: InceptionResNet (mixed)  ")
    print("====================================\n")

    inc_model = build_model("INCEPTION_RESNET", vocab_size)
    inc_history, inc_train_acc, inc_test_acc = train_and_evaluate(
        inc_model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="InceptionResNet_Model",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n[ InceptionResNet ] Computing confusion matrix...")
    inc_probs = inc_model.predict(X_test, batch_size=args.batch_size).ravel()
    inc_preds = (inc_probs >= 0.5).astype(int)
    plot_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=inc_preds,
        model_name="InceptionResNet_Model",
        save_dir="outputs",
    )
    print("[ InceptionResNet ] Confusion matrix saved to outputs/InceptionResNet_Model_cm.png")

    print("\n[ InceptionResNet ] Classification report (0 = Real, 1 = Fake):")
    print(
        classification_report(
            y_test,
            inc_preds,
            target_names=["Fake (0)", "Real (1)"],
            digits=4,
        )
    )

    print("[ InceptionResNet ] Saving full model + tokenizer...")
    save_artifacts(inc_model, dm.tokenizer, model_name="inception_resnet_model")

    add_metrics_rows("InceptionResNet", y_test, inc_preds, inc_test_acc)

    # ============================================================
    # 5) Train BERT (PyTorch) — labels: 0 = Real, 1 = Fake
    # ============================================================
    print("\n=====================")
    print(" TRAINING: BERT Model")
    print("=====================\n")

    (
        bert_model,
        bert_tokenizer,
        bert_train_acc,
        bert_val_acc,
        bert_train_size,
        bert_val_size,
        bert_val_labels,
        bert_val_preds,
    ) = train_bert_welfake(
        max_samples=None,          # 👉 None = use ALL samples
        model_name="bert-base-uncased",
        max_len=160,
        batch_size=8,
        epochs=args.bert_epochs,
    )

    # Add BERT metrics (validation as test set)
    add_metrics_rows("BERT", bert_val_labels, bert_val_preds, bert_val_acc)

    print(
        "\n[BERT] Training complete. "
        "Check outputs/BERT_Model_cm.png for confusion matrix, "
        "and models/ for welfake_bert_* artifacts."
    )

    # ============================================================
    # 6) Save metrics table for ALL models
    # ============================================================
    print("\nSaving metrics table for all models...")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join("outputs", "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Metrics (precision/recall/F1/accuracy) saved to: {metrics_csv_path}")

    # ============================================================
    # 7) Summary
    # ============================================================
    print("\n========================")
    print(" TRAINING SUMMARY (ALL) ")
    print("========================")
    print("Data split for TF models (CNN/LSTM/InceptionResNet): 80% train / 20% test.")
    print(
        f"Data split for BERT: {bert_train_size} training samples "
        f"and {bert_val_size} validation (testing) samples (~80% / ~20%)."
    )

    print("\n[ CNN_Model ]")
    print(f"  Final Training Accuracy: {cnn_train_acc:.4f}")
    print(f"  Test Accuracy          : {cnn_test_acc:.4f}")

    print("\n[ LSTM_Model ]")
    print(f"  Final Training Accuracy: {lstm_train_acc:.4f}")
    print(f"  Test Accuracy          : {lstm_test_acc:.4f}")

    print("\n[ InceptionResNet_Model ]")
    print(f"  Final Training Accuracy: {inc_train_acc:.4f}")
    print(f"  Test Accuracy          : {inc_test_acc:.4f}")

    print("\n[ BERT_Model ]")
    print(f"  Final Training Accuracy        : {bert_train_acc:.4f}")
    print(f"  Final Validation/Test Accuracy : {bert_val_acc:.4f}")

    print("\n(✅ All TF models and BERT have been trained, evaluated, and saved.)")
    print("Check the outputs/ folder for curves, confusion matrices, and CSV reports. 🎉🔥")


if __name__ == "__main__":
    main()
