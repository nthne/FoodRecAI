import argparse
from html import escape
from pathlib import Path
import sys

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .dataset import LABELS, load_data_splits
    from .train import DEFAULT_MODEL_PATH, train_model
except ImportError:
    from dataset import LABELS, load_data_splits
    from train import DEFAULT_MODEL_PATH, train_model


OUTPUT_DIR = Path(__file__).resolve().parent / "result"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def write_report(output_path, lines):
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(data_dir=None, model_path=DEFAULT_MODEL_PATH, train_if_missing=False):
    model_path = Path(model_path)
    if not model_path.exists():
        if not train_if_missing:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run train.py first or pass --train-if-missing."
            )
        train_model(data_dir=data_dir, model_path=model_path)

    _, _, x_test, _, _, y_test, stats = load_data_splits(data_dir=data_dir)
    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    return metrics, report, matrix, stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate Logistic Regression model.")
    parser.add_argument(
        "--data-dir",
        "--data",
        dest="data_dir",
        help="Directory containing train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = Path(args.model)
    metrics, report, matrix, stats = evaluate_model(
        data_dir=args.data_dir,
        model_path=model_path,
        train_if_missing=args.train_if_missing,
    )

    report_lines = [
        "===== DATA =====",
        f"Data dir: {stats['data_dir']}",
        f"Train file: {stats['train_path']}",
        f"Valid file: {stats['valid_path']}",
        f"Test file: {stats['test_path']}",
        f"Total used: {stats['total_used']}",
        f"Train size: {stats['train_size']}",
        f"Valid size: {stats['valid_size']}",
        f"Test size: {stats['test_size']}",
        f"Skipped invalid rows: {stats['skipped_invalid']}",
        f"Model file: {model_path}",
        "",
        "===== METRICS =====",
        *[f"{name}: {value:.4f}" for name, value in metrics.items()],
        "",
        "===== CLASSIFICATION REPORT =====",
        report,
        "===== CONFUSION MATRIX =====",
        str(matrix),
    ]

    report_path = OUTPUT_DIR / "logistic_regression_report.txt"
    matrix_path = OUTPUT_DIR / "confusion_matrix_3_labels.svg"

    write_report(report_path, report_lines)
    matrix_path.write_text(
        matrix_to_svg(matrix, "Logistic Regression Sentiment Confusion Matrix"),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nSaved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


def matrix_to_svg(matrix, title):
    max_value = max(matrix.max(), 1)
    cell_size = 110
    label_size = 120
    title_height = 60
    width = label_size + cell_size * len(LABELS) + 30
    height = title_height + label_size + cell_size * len(LABELS) + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" '
        f'font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<text x="{label_size + cell_size * len(LABELS) / 2}" y="{title_height + 30}" '
        'text-anchor="middle" font-family="Arial" font-size="14">Predicted label</text>',
        f'<text x="20" y="{title_height + label_size + cell_size * len(LABELS) / 2}" '
        'text-anchor="middle" font-family="Arial" font-size="14" '
        f'transform="rotate(-90 20 {title_height + label_size + cell_size * len(LABELS) / 2})">True label</text>',
    ]

    for col, label in enumerate(LABELS):
        x = label_size + col * cell_size + cell_size / 2
        y = title_height + label_size - 20
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial" '
            f'font-size="13">{escape(label)}</text>'
        )

    for row, label in enumerate(LABELS):
        x = label_size - 15
        y = title_height + label_size + row * cell_size + cell_size / 2 + 5
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="end" font-family="Arial" '
            f'font-size="13">{escape(label)}</text>'
        )

    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            value = int(matrix[row, col])
            intensity = value / max_value
            blue = int(245 - 150 * intensity)
            fill = f"rgb({blue},{blue + 5},255)"
            x = label_size + col * cell_size
            y = title_height + label_size + row * cell_size
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="#d0d7de"/>'
            )
            parts.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 6}" '
                'text-anchor="middle" font-family="Arial" font-size="18" '
                f'font-weight="700" fill="#111827">{value}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
