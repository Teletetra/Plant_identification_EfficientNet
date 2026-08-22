import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image

from src.model.cnn import (
    CBAM,
    ChannelAttention,
    SpatialAttention
)

from src.utils.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE
)


class PlantPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH
    ):

        model_path = Path(model_path)

        class_names_path = Path(
            class_names_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not class_names_path.exists():

            raise FileNotFoundError(
                f"Class names not found: "
                f"{class_names_path}"
            )

        self.model = tf.keras.models.load_model(

            model_path,

            custom_objects={
                "CBAM": CBAM,
                "ChannelAttention": ChannelAttention,
                "SpatialAttention": SpatialAttention
            }
        )

        self.class_names = json.loads(
            class_names_path.read_text()
        )

    def predict(
        self,
        image: Image.Image
    ):

        image = image.convert("RGB")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        probabilities = self.model.predict(
            image_array,
            verbose=0
        )[0]

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index]
        )

        return {
            "plant": self.class_names[
                class_index
            ],

            "confidence": confidence,

            "class_index": class_index
        }