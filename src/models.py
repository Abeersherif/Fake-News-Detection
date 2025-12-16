# src/models.py

import tensorflow as tf
from tensorflow.keras import layers, models


def _make_input(max_len: int | None = None):
    """
    Helper to create the Input layer.

    If max_len is None -> variable-length sequences.
    If max_len is int  -> fixed-length sequences (e.g. 200).
    """
    if max_len is None:
        return layers.Input(shape=(None,), dtype="int32", name="input_ids")
    else:
        return layers.Input(shape=(max_len,), dtype="int32", name="input_ids")


# ============================================================
# 1) CNN model  (Embedding FROZEN = non-trainable)
# ============================================================
def build_cnn_model(
    vocab_size: int,
    max_len: int | None = None,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    pretrained: bool = False,  # Add flag to control freezing
) -> tf.keras.Model:
    """
    Simple 1D CNN for text classification (binary: fake / real).
    Embedding layer is frozen (non-trainable) if pretrained is True.
    """
    inputs = _make_input(max_len)

    # If pretrained, freeze embedding weights
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
        trainable=not pretrained,  # Freeze embedding if pretrained
    )(inputs)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        name="conv1d_1",
    )(x)

    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        name="conv1d_2",
    )(x)

    x = layers.GlobalMaxPooling1D(name="global_maxpool")(x)

    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Model")
    return model


# ============================================================
# 2) LSTM model  (Embedding FROZEN)
# ============================================================
def build_lstm_model(
    vocab_size: int,
    max_len: int | None = None,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    dense_units: int = 64,
    pretrained: bool = False,  # Add flag to control freezing
) -> tf.keras.Model:
    """
    Simple LSTM model for text classification (binary: fake / real).
    Embedding layer is frozen (non-trainable) if pretrained is True.
    """
    inputs = _make_input(max_len)

    # If pretrained, freeze embedding weights
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
        trainable=not pretrained,  # Freeze embedding if pretrained
    )(inputs)

    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.5, name="dropout2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_Model")
    return model


# ============================================================
# 3) Inception + Residual (mixed) model  (Embedding FROZEN)
# ============================================================
def build_inception_resnet_model(
    vocab_size: int,
    max_len: int | None = None,
    embedding_dim: int = 128,
    pretrained: bool = False,  # Add flag to control freezing
) -> tf.keras.Model:
    """
    Inception-like parallel Conv1D branches + Residual connection.
    Embedding layer is frozen (non-trainable) if pretrained is True.
    """
    inputs = _make_input(max_len)

    # If pretrained, freeze embedding weights
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
        trainable=not pretrained,  # Freeze embedding if pretrained
    )(inputs)

    # Parallel branches
    b1 = layers.Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="inc_branch_3",
    )(x)
    b2 = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="inc_branch_5",
    )(x)
    b3 = layers.Conv1D(
        64,
        kernel_size=7,
        padding="same",
        activation="relu",
        name="inc_branch_7",
    )(x)

    inception = layers.Concatenate(
        axis=-1,
        name="inception_concat",
    )([b1, b2, b3])  # shape: (..., 64*3)

    # Residual path (1x1 conv to match channels)
    res = layers.Conv1D(
        filters=inception.shape[-1],
        kernel_size=1,
        padding="same",
        name="res_1x1",
    )(x)

    x = layers.Add(name="inception_res_add")([inception, res])
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    x = layers.GlobalMaxPooling1D(name="global_maxpool")(x)
    x = layers.Dense(128, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="InceptionResNet_Model")
    return model


# ============================================================
# 4) Factory function used by main.py (with pretrained flag)
# ============================================================
def build_model(
    model_type: str,
    vocab_size: int,
    max_len: int | None = None,
    embed_dim: int = 128,
    pretrained: bool = False,  # Add flag to control frozen layers
) -> tf.keras.Model:
    """
    Factory function used by main.py

    model_type: "CNN", "LSTM", or "INCEPTION_RESNET"
    pretrained: Whether to freeze early layers of the model.
    """
    model_type_upper = model_type.upper()

    if model_type_upper == "CNN":
        model = build_cnn_model(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=embed_dim,
            pretrained=pretrained,
        )
    elif model_type_upper == "LSTM":
        model = build_lstm_model(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=embed_dim,
            pretrained=pretrained,
        )
    elif model_type_upper in ["INCEPTION_RESNET", "INCEPTION_RESIDUAL"]:
        model = build_inception_resnet_model(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=embed_dim,
            pretrained=pretrained,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Use 'CNN', 'LSTM', or 'INCEPTION_RESNET'."
        )

    print(
        f"[build_model] Built {model.name} "
        f"(type={model_type_upper}) with vocab_size={vocab_size}, max_len={max_len}, pretrained={pretrained}"
    )
    model.summary()
    return model
