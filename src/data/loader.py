from pathlib import Path
import tensorflow as tf

from src.utils.config import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(directory: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )

def prepare_datasets(processed_dir: Path):
    train_ds = create_dataset(processed_dir / "train", shuffle=True)
    val_ds = create_dataset(processed_dir / "validation", shuffle=False)
    test_ds = create_dataset(processed_dir / "test", shuffle=False)

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
