import tensorflow as tf

from src.model.cnn import build_cnn

def test_model_output_shape():
    model = build_cnn(num_classes=4, input_shape=(224, 224, 3))
    assert model.output_shape == (None, 4)

def test_model_forward_pass():
    model = build_cnn(num_classes=3, input_shape=(224, 224, 3))
    x = tf.random.uniform((2, 224, 224, 3))
    y = model(x, training=False)
    assert y.shape == (2, 3)

