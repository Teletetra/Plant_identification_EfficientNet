from pathlib import Path
import random
import shutil

from src.utils.config import RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(directory: Path):
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def prepare_dataset(raw_dir: Path, output_dir: Path):
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")

    classes = sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {raw_dir}")

    random.seed(RANDOM_SEED)

    # Clean old processed data.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    counts = {"train": {}, "validation": {}, "test": {}}

    for class_name in classes:
        images = list_images(raw_dir / class_name)
        random.shuffle(images)

        n = len(images)
        test_n = max(1, int(n * TEST_SIZE)) if n >= 3 else 0
        val_n = max(1, int(n * VALIDATION_SIZE)) if n >= 3 else 0

        test_images = images[:test_n]
        val_images = images[test_n:test_n + val_n]
        train_images = images[test_n + val_n:]

        if not train_images and images:
            train_images = images[:1]

        splits = {
            "train": train_images,
            "validation": val_images,
            "test": test_images,
        }

        for split, split_images in splits.items():
            destination = output_dir / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            counts[split][class_name] = len(split_images)

            for index, source in enumerate(split_images):
                # Prefix prevents filename collisions.
                target = destination / f"{index:06d}_{source.name}"
                shutil.copy2(source, target)

    return classes, counts
