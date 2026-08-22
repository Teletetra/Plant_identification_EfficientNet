import json

from src.data.preprocessing import prepare_dataset
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR

if __name__ == "__main__":
    classes, counts = prepare_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    info = {
        "classes": classes,
        "splits": counts,
    }
    (DATA_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print("Classes:", classes)
    print("Dataset counts:")
    for split, split_counts in counts.items():
        print(split, split_counts)
