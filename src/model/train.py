import json
import tensorflow as tf

from src.data.loader import prepare_datasets
from src.model.cnn import build_cnn
from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    TRAINING_HISTORY_PATH,
    IMAGE_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

def train():
    train_ds, val_ds, _, class_names = prepare_datasets(PROCESSED_DATA_DIR)

    model = build_cnn(
        num_classes=len(class_names),
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    TRAINING_HISTORY_PATH.write_text(json.dumps(history.history, indent=2))

    return model, history, class_names
