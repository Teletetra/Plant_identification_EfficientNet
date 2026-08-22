import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# CBAM: Convolutional Block Attention Module
# ============================================================

class ChannelAttention(layers.Layer):
    """
    Channel attention:
    Learns which feature channels are important.
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden_channels = max(channels // self.reduction_ratio, 1)

        self.shared_dense_1 = layers.Dense(
            hidden_channels,
            activation="relu",
            use_bias=True
        )

        self.shared_dense_2 = layers.Dense(
            channels,
            use_bias=True
        )

        super().build(input_shape)

    def call(self, inputs):
        # Average pooling across spatial dimensions
        avg_pool = tf.reduce_mean(
            inputs,
            axis=[1, 2],
            keepdims=False
        )

        # Max pooling across spatial dimensions
        max_pool = tf.reduce_max(
            inputs,
            axis=[1, 2],
            keepdims=False
        )

        avg_attention = self.shared_dense_2(
            self.shared_dense_1(avg_pool)
        )

        max_attention = self.shared_dense_2(
            self.shared_dense_1(max_pool)
        )

        attention = avg_attention + max_attention
        attention = tf.nn.sigmoid(attention)

        attention = tf.reshape(
            attention,
            [-1, 1, 1, tf.shape(attention)[-1]]
        )

        return inputs * attention


class SpatialAttention(layers.Layer):
    """
    Spatial attention:
    Learns which spatial regions of the image are important.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False
        )

    def call(self, inputs):
        # Average across channels
        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        # Max across channels
        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        combined = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        attention = self.conv(combined)

        return inputs * attention


class CBAM(layers.Layer):
    """
    Complete CBAM module:
    
    Feature Map
         ↓
    Channel Attention
         ↓
    Spatial Attention
         ↓
    Refined Feature Map
    """

    def __init__(
        self,
        reduction_ratio=16,
        spatial_kernel_size=7,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size
        )

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)

        return x


# ============================================================
# Data Augmentation
# ============================================================

def build_augmentation():

    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal"
            ),

            layers.RandomRotation(
                factor=0.10
            ),

            layers.RandomZoom(
                height_factor=0.15,
                width_factor=0.15
            ),

            layers.RandomTranslation(
                height_factor=0.10,
                width_factor=0.10
            ),

            layers.RandomContrast(
                factor=0.15
            ),
        ],
        name="plant_augmentation"
    )


# ============================================================
# EfficientNet + CBAM Model
# ============================================================

def build_model(
    num_classes: int,
    input_shape=(224, 224, 3),
    weights="imagenet"
):

    inputs = layers.Input(
        shape=input_shape,
        name="plant_image"
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    x = build_augmentation()(inputs)

    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )

    # Freeze backbone initially
    backbone.trainable = False

    x = backbone(
        x,
        training=False
    )

    # --------------------------------------------------------
    # CBAM Attention
    # --------------------------------------------------------

    x = CBAM(
        reduction_ratio=16,
        spatial_kernel_size=7,
        name="cbam"
    )(x)

    # --------------------------------------------------------
    # Global feature extraction
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = layers.Dropout(
        0.40,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="plant_class"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_CBAM_PlantClassifier"
    )

    return model, backbone


# ============================================================
# Compile model for Stage 1
# ============================================================

def compile_stage_1(
    model,
    learning_rate=1e-3
):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model


# ============================================================
# Prepare model for Stage 2 Fine-Tuning
# ============================================================

def prepare_fine_tuning(
    model,
    backbone,
    unfreeze_layers=30
):

    # First freeze everything
    backbone.trainable = True

    # Freeze all layers except last N layers
    total_layers = len(backbone.layers)

    freeze_until = max(
        total_layers - unfreeze_layers,
        0
    )

    for layer in backbone.layers[:freeze_until]:

        layer.trainable = False

    # Keep BatchNorm layers frozen during fine-tuning
    for layer in backbone.layers:

        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

    return model


# ============================================================
# Compile Stage 2
# ============================================================

def compile_stage_2(
    model,
    learning_rate=1e-5
):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model