import json

import tensorflow as tf

from src.data.loader import prepare_datasets
from src.model.cnn import (
    build_model,
    compile_stage_1,
    prepare_fine_tuning,
    compile_stage_2,
)

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    TRAINING_HISTORY_PATH,
    IMAGE_SIZE,
)


STAGE_1_EPOCHS = 8
STAGE_2_EPOCHS = 15

STAGE_1_LR = 1e-3
STAGE_2_LR = 1e-5


def create_callbacks():

    return [

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


def train():

    print("\nLoading datasets...")

    train_ds, val_ds, test_ds, class_names = (
        prepare_datasets(
            PROCESSED_DATA_DIR
        )
    )

    num_classes = len(class_names)

    print("\nClasses:")
    for index, class_name in enumerate(class_names):
        print(
            f"{index}: {class_name}"
        )

    print(
        f"\nNumber of classes: {num_classes}"
    )

    # ========================================================
    # Build Model
    # ========================================================

    print("\nBuilding EfficientNet-B0 + CBAM...")

    model, backbone = build_model(
        num_classes=num_classes,
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    )

    model.summary()

    # ========================================================
    # STAGE 1
    # Feature Extraction
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 1: FEATURE EXTRACTION")
    print("=" * 60)

    compile_stage_1(
        model,
        learning_rate=STAGE_1_LR
    )

    history_stage_1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=STAGE_1_EPOCHS,

        callbacks=create_callbacks()
    )

    # ========================================================
    # STAGE 2
    # Fine Tuning
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 2: FINE-TUNING")
    print("=" * 60)

    prepare_fine_tuning(
        model,
        backbone,
        unfreeze_layers=30
    )

    compile_stage_2(
        model,
        learning_rate=STAGE_2_LR
    )

    history_stage_2 = model.fit(

        train_ds,

        validation_data=val_ds,

        initial_epoch=STAGE_1_EPOCHS,

        epochs=(
            STAGE_1_EPOCHS
            + STAGE_2_EPOCHS
        ),

        callbacks=create_callbacks()
    )

    # ========================================================
    # Save Model
    # ========================================================

    model.save(MODEL_PATH)

    # ========================================================
    # Save Classes
    # ========================================================

    CLASS_NAMES_PATH.write_text(
        json.dumps(
            class_names,
            indent=2
        )
    )

    # ========================================================
    # Combine Training History
    # ========================================================

    history = {}

    for key in history_stage_1.history:

        history[key] = (
            history_stage_1.history[key]
            + history_stage_2.history.get(
                key,
                []
            )
        )

    TRAINING_HISTORY_PATH.write_text(
        json.dumps(
            history,
            indent=2
        )
    )

    print("\nTraining completed.")

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    return (
        model,
        history,
        class_names
    )