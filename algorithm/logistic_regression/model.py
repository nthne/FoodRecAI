from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from .dataset import LABELS


def build_logistic_pipeline(
    num_classes=None,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    random_state=42,
):
    
    if num_classes is None:
        num_classes = len(LABELS)

    if num_classes < 2:
        raise ValueError("Logistic Regression needs at least 2 classes to train.")

    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )

    return Pipeline(steps=[("tfidf", tfidf), ("classifier", clf)])


def train_pipeline(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def evaluate_model(pipeline, X_test, y_test, labels=None):
    if labels is None:
        labels = LABELS

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=labels)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return report, conf_matrix

