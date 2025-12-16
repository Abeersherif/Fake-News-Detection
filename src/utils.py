import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_history(history, model_name: str, save_dir: str = "outputs") -> None:
    """
    Plot training & validation accuracy and loss, then save the figure as:
        {save_dir}/{model_name}_history.png
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training acc")
    plt.plot(epochs, val_acc, "ro-", label="Validation acc")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    out_path = os.path.join(save_dir, f"{model_name}_history.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_history] Saved to {out_path}")


def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    model_name: str,
    save_dir: str = "outputs",
    labels: tuple[str, str] = ("Fake", "Real"),
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Assumes:
      0 = Fake
      1 = Real

    We fix the label order as [0, 1] to keep mapping consistent.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Fix label order: [0, 1] → ['Fake', 'Real']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    out_path = os.path.join(save_dir, f"{model_name}_cm.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_confusion_matrix_heatmap] Saved to {out_path}")


def save_artifacts(model, tokenizer, model_name: str = "best_model") -> None:
    """
    Save trained Keras model and tokenizer under models/ folder.

    Example:
      model_name='cnn_model'  →  models/cnn_model.h5
      tokenizer               →  models/tokenizer.pickle  (shared)
    """
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", f"{model_name}.h5")
    tok_path = os.path.join("models", "tokenizer.pickle")

    print(f"[save_artifacts] Saving model to {model_path}")
    model.save(model_path)

    print(f"[save_artifacts] Saving tokenizer to {tok_path}")
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print("[save_artifacts] Done.")
