import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .dataset import LABELS, load_data_splits
    from .model import build_logistic_pipeline
except ImportError:
    from dataset import LABELS, load_data_splits
    from model import build_logistic_pipeline


OUTPUT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "logistic_regression_model.joblib"


def evaluate_split(model, x, y_true, labels):
    y_pred = model.predict(x)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_model(
    data_dir=None,
    output_dir=None,
    model_path=None,
    max_features=50000,
    ngram_max=2,
    min_df=2,
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    random_state=42,
):
    x_train, x_valid, x_test, y_train, y_valid, y_test, data_stats = load_data_splits(data_dir=data_dir)

    class_names = [label for label in LABELS if label in set(y_train + y_valid + y_test)]
    if len(set(y_train)) < 2:
        raise ValueError("Training split must contain at least 2 classes.")

    model = build_logistic_pipeline(
        num_classes=len(set(y_train)),
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path) if model_path else output_dir / DEFAULT_MODEL_PATH.name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    valid_metrics = evaluate_split(model, x_valid, y_valid, labels=class_names)
    test_metrics = evaluate_split(model, x_test, y_test, labels=class_names)

    metadata = {
        "model": "TF-IDF + Logistic Regression",
        "class_names": class_names,
        "data_stats": data_stats,
        "train_label_distribution": dict(Counter(y_train)),
        "valid_label_distribution": dict(Counter(y_valid)),
        "test_label_distribution": dict(Counter(y_test)),
        "params": {
            "max_features": max_features,
            "ngram_max": ngram_max,
            "min_df": min_df,
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state,
        },
        "model_path": str(model_path),
    }

    with open(output_dir / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"valid": valid_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)

    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + class_names)
        for label, row in zip(class_names, test_metrics["confusion_matrix"]):
            writer.writerow([label] + row)

    stats = {
        **data_stats,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "valid_metrics": {
            "accuracy": valid_metrics["accuracy"],
            "macro_f1": valid_metrics["macro_f1"],
            "weighted_f1": valid_metrics["weighted_f1"],
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }

    return model, stats


class WrappedModel:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, X):
        X_t = self.vectorizer.transform(X)
        return self.classifier.predict(X_t)


def run_selection(
    data_dir=None,
    output_dir=None,
    model_path=None,
    rounds=4,
    sample_frac=1.0,
    max_features=50000,
    ngram_max=2,
    min_df=2,
    C=1.0,
    random_state=42,
):
    x_train, x_valid, x_test, y_train, y_valid, y_test, data_stats = load_data_splits(data_dir=data_dir)

    rng = random.Random(random_state)

    vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, ngram_max), max_features=max_features, min_df=min_df)
    vectorizer.fit(x_train + x_valid)

    tfidf_valid = vectorizer.transform(x_valid)
    valid_centroid = np.asarray(tfidf_valid.mean(axis=0)).ravel()

    best_sim = -1.0
    best_model = None
    best_stats = None

    n_train = len(x_train)
    sample_size = int(round(n_train * sample_frac))

    for r in range(rounds):
        sampled_idx = [rng.randrange(n_train) for _ in range(sample_size)]
        sampled_x = [x_train[i] for i in sampled_idx]
        sampled_y = [y_train[i] for i in sampled_idx]

        tfidf_sampled = vectorizer.transform(sampled_x)
        train_centroid = np.asarray(tfidf_sampled.mean(axis=0)).ravel()
        sim = float(cosine_similarity(train_centroid.reshape(1, -1), valid_centroid.reshape(1, -1)).ravel()[0])

        clf = LogisticRegression(C=C, penalty="l2", solver="saga", max_iter=1000, random_state=random_state + r)
        clf.fit(tfidf_sampled, sampled_y)

        wrapped = WrappedModel(vectorizer, clf)

        test_metrics = evaluate_split(wrapped, x_test, y_test, labels=[label for label in LABELS if label in set(y_train + y_valid + y_test)])

        if sim > best_sim:
            best_sim = sim
            best_model = wrapped
            best_stats = {
                "round": r,
                "similarity": sim,
                "sample_size": sample_size,
                "test_metrics": test_metrics,
            }

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path) if model_path else output_dir / DEFAULT_MODEL_PATH.name
    joblib.dump(best_model, model_path)

    metadata = {
        "method": "bootstrap-sample-select",
        "rounds": rounds,
        "best_similarity": best_sim,
        "best_round": best_stats["round"],
        "model_path": str(model_path),
        "data_stats": data_stats,
    }

    with open(output_dir / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": best_stats["test_metrics"]}, f, ensure_ascii=False, indent=2)

    import csv
    class_names = best_stats["test_metrics"]["classification_report"].keys()
    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + list(class_names))
        for label, row in zip(list(class_names), best_stats["test_metrics"]["confusion_matrix"]):
            writer.writerow([label] + row)

    return model_path, metadata


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression sentiment model.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.jsonl, valid.jsonl, test.jsonl. Default: data/data_for_sentiment_analysis",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", default="l2", choices=["l1", "l2", "elasticnet", "none"])
    parser.add_argument("--solver", default="saga", choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--select-rounds", type=int, default=0, help="If >0, run selection over N bootstrap rounds and pick best model by TF-IDF centroid similarity to valid")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of train to sample for each round (bootstrap) when using --select-rounds")
    args = parser.parse_args()

    if args.select_rounds and args.select_rounds > 0:
        model_path, metadata = run_selection(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_path=args.model_out,
            rounds=args.select_rounds,
            sample_frac=args.sample_frac,
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            C=args.C,
            random_state=args.random_state,
        )

        print("Selection completed.")
        print(f"Saved model: {model_path}")
        print(f"Best similarity: {metadata.get('best_similarity')}")
        return

    _, stats = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_out,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    print("Training completed.")
    print(f"Data dir: {stats['data_dir']}")
    print(f"Train file: {stats['train_path']}")
    print(f"Valid file: {stats['valid_path']}")
    print(f"Test file: {stats['test_path']}")
    print(f"Saved model: {stats['model_path']}")
    print("Validation metrics:")
    print(f"  Accuracy: {stats['valid_metrics']['accuracy']:.4f}")
    print(f"  Macro F1: {stats['valid_metrics']['macro_f1']:.4f}")
    print(f"  Weighted F1: {stats['valid_metrics']['weighted_f1']:.4f}")
    print("Test metrics:")
    print(f"  Accuracy: {stats['test_metrics']['accuracy']:.4f}")
    print(f"  Macro F1: {stats['test_metrics']['macro_f1']:.4f}")
    print(f"  Weighted F1: {stats['test_metrics']['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
