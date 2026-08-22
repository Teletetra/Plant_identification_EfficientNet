import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.utils.config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

class PlantPredictor:
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        model_path = Path(model_path)
        class_names_path = Path(class_names_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train the model first."
            )

        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Class names not found: {class_names_path}. Train the model first."
            )

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = json.loads(class_names_path.read_text())

    def predict(self, image: Image.Image):
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probabilities = self.model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))

        return {
            "plant": self.class_names[index],
            "confidence": float(probabilities[index]),
            "class_index": index,
        }
