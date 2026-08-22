import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import prepare_datasets
from src.utils.config import PROCESSED_DATA_DIR, MODEL_PATH, MODEL_DIR

def evaluate():
    _, _, test_ds, class_names = prepare_datasets(PROCESSED_DATA_DIR)

    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(test_ds, verbose=1)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    (MODEL_DIR / "classification_report.json").write_text(
        json.dumps(result, indent=2)
    )

    return result
