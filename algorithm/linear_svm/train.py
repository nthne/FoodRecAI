import argparse
from pathlib import Path
import joblib
from data_processor import load_split_data
from model import build_svm_pipeline
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def train_model(data_dir, random_state, c_param, max_features, version_name):
    # Data splitting 
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
        data_dir=data_dir)
    
    # Build SVM pipeline
    pipeline = build_svm_pipeline(max_features=max_features, c_param=c_param, random_state=random_state)
    print(f"-> Training Processing ... (C={c_param}, Max_Features={max_features}) for the version: {version_name}...")
    
    # Train model
    pipeline.fit(X_train, y_train)
    print("-> Training Partially Successfully")
    
    # Evaluate on validation set
    val_accuracy = pipeline.score(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Combine Train and Valid sets for final training to maximize data usage
    print(" -> Re-training on combined Train + Valid set for final model ...")
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    
    # Re-train the pipeline on the combined dataset
    pipeline.fit(X_train_full, y_train_full)
    print("-> Final Training Successfully Completed")

    # Save the trained pipeline to disk
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / f'svm_pipeline_{version_name}.pkl'
    joblib.dump(pipeline, model_path)
    print(f"-> Optimal pipeline saved to {model_path}")
    
    # Calculate stats for report
    stats = {
        "data_dir": str(Path(data_dir).resolve() if data_dir else Path(__file__).resolve().parent),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "total_used": len(X_train) + len(X_val) + len(X_test),
        "model_path": str(model_path.resolve()),
        "c_param": c_param,
        "max_features": max_features,
        "version": version_name,
        "val_acc": val_accuracy
    }
    
    return pipeline, stats


def main():
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Train an SVM model for text classification with specified parameters and data directory.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--version", default="C0.1")
    args = parser.parse_args()

    # Train the model and get stats for reporting
    _, stats = train_model(
        data_dir=args.data_dir, 
        random_state=args.random_state, 
        c_param=args.c, 
        max_features=args.max_features, 
        version_name=args.version
    )

    # Print the training report
    print("\n" + "="*20 + " TRAIN REPORT " + "="*20)
    print(f"Data Directory:  {stats['data_dir']}")
    print(f"Total Dataset:   {stats['total_used']} rows")
    print(f"├── Train size:  {stats['train_size']} rows")
    print(f"├── Valid size:  {stats['val_size']} rows")
    print(f"└── Test size:   {stats['test_size']} rows")
    print("-" * 54)
    print(f"SVM Param C:     {stats['c_param']}")
    print(f"TFIDF Max Feat:  {stats['max_features']}")
    print(f"Validation Acc:  {stats['val_acc'] * 100:.2f}%")
    print(f"Saved Version:   {stats['version']}")
    print(f"Model Path:      {stats['model_path']}")
    print("="*54 + "\n")

if __name__ == "__main__":
    main()
