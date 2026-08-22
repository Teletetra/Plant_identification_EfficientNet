# Plant Identification CNN

Production-style plant image classification project using TensorFlow/Keras, FastAPI, and a lightweight HTML/CSS/JS frontend.

## Pipeline

Raw images → preprocessing → CNN training → evaluation → saved Keras model → inference → FastAPI → frontend.

## Dataset structure

Place images in:

```text
data/raw/plant_images/
├── apple/
├── potato/
├── tomato/
└── ...
```

Each folder name becomes a class.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust values if required.

## Prepare dataset

```bash
python scripts/prepare_dataset.py
```

This creates `data/processed/train`, `validation`, and `test`.

## Train

```bash
python scripts/train_model.py
```

## Evaluate

```bash
python scripts/evaluate_model.py
```

## Run API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

For the frontend, open `frontend/index.html` through a local static server, for example:

```bash
python -m http.server 5500 --directory frontend
```

Then open `http://127.0.0.1:5500`.

## Notes

- Raw and processed datasets are ignored by Git.
- Large `.keras` model files are ignored by default.
- The API expects a trained model at `models/plant_classifier.keras`.
