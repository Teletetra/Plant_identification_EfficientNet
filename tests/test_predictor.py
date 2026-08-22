import json

import numpy as np
import tensorflow as tf
from PIL import Image

from src.model.cnn import build_cnn
from src.inference.predictor import PlantPredictor

def test_predictor(tmp_path):
    model_path = tmp_path / "model.keras"
    classes_path = tmp_path / "classes.json"

    model = build_cnn(num_classes=2, input_shape=(224, 224, 3))
    model.save(model_path)
    classes_path.write_text(json.dumps(["apple", "tomato"]))

    predictor = PlantPredictor(model_path, classes_path)

    image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    result = predictor.predict(image)

    assert result["plant"] in ["apple", "tomato"]
    assert 0 <= result["confidence"] <= 1
    assert result["class_index"] in [0, 1]
