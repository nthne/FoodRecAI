import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_for_sentiment_analysis"
DEFAULT_SPLIT_PATHS = {
    "train": DEFAULT_DATA_DIR / "train.jsonl",
    "valid": DEFAULT_DATA_DIR / "valid.jsonl",
    "test": DEFAULT_DATA_DIR / "test.jsonl",
}
LABELS = ["negative", "neutral", "positive"]

def rating_to_label(rating):
    if rating >= 8.0:
        return "positive"
    elif rating >= 4.0:
        return "neutral"
    return "negative"


def resolve_data_dir(data_dir=None):
    if data_dir:
        path = Path(data_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return DEFAULT_DATA_DIR


def resolve_split_path(split, data_dir=None):
    if split not in DEFAULT_SPLIT_PATHS:
        raise ValueError(f"Unknown split: {split}")

    if data_dir:
        path = Path(data_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path / f"{split}.jsonl"

    return DEFAULT_SPLIT_PATHS[split]


def load_reviews(split, data_dir=None):
    path = resolve_split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    texts = []
    labels = []
    skipped_invalid = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            rating = item.get("rating")
            if rating is not None:
                try:
                    rating = float(rating)
                except (TypeError, ValueError):
                    skipped_invalid += 1
                    continue

            title = item.get("title") or ""
            content = item.get("content") or ""
            text = f"{title} {content}".strip()
            if not text:
                skipped_invalid += 1
                continue

            label = item.get("rating_label")
            if not label and rating is not None:
                label = rating_to_label(rating)
            if label not in LABELS:
                skipped_invalid += 1
                continue

            texts.append(text)
            labels.append(label)

    stats = {
        "data_path": str(path),
        "split": split,
        "total_used": len(texts),
        "skipped_invalid": skipped_invalid,
    }
    return texts, labels, stats


def load_data_splits(data_dir=None):
    x_train, y_train, train_stats = load_reviews("train", data_dir=data_dir)
    x_valid, y_valid, valid_stats = load_reviews("valid", data_dir=data_dir)
    x_test, y_test, test_stats = load_reviews("test", data_dir=data_dir)

    if not x_train:
        raise ValueError("No valid train reviews.")
    if not x_valid:
        raise ValueError("No valid validation reviews.")
    if not x_test:
        raise ValueError("No valid test reviews.")

    stats = {
        "data_dir": str(resolve_data_dir(data_dir)),
        "train_path": train_stats["data_path"],
        "valid_path": valid_stats["data_path"],
        "test_path": test_stats["data_path"],
        "train_size": len(x_train),
        "valid_size": len(x_valid),
        "test_size": len(x_test),
        "total_used": len(x_train) + len(x_valid) + len(x_test),
        "skipped_invalid": (
            train_stats["skipped_invalid"]
            + valid_stats["skipped_invalid"]
            + test_stats["skipped_invalid"]
        ),
    }
    return x_train, x_valid, x_test, y_train, y_valid, y_test, stats