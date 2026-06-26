import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

MODEL_PATH = Path("model.pkl")


def resolve_train_csv() -> Path:
    # Place train.csv dataset in the root directory.
    for candidate in (Path("train.csv"), Path("data") / "train.csv"):
        if candidate.exists():
            print(f"Using local dataset: {candidate}")
            return candidate

    raise FileNotFoundError(
        "train.csv not found. Place the dataset at ./train.csv or ./data/train.csv."
    )


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"@(\w+)", r" \1 ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_input(text: str, keyword: str = "") -> str:
    # Combine the predictive `keyword` column with the tweet text.
    if not isinstance(keyword, str):
        keyword = ""
    keyword = keyword.replace("%20", " ")
    kw_clean = clean_text(keyword)
    txt_clean = clean_text(text)
    return (kw_clean + " " + txt_clean).strip()


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    max_features=50000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    logreg = LogisticRegression(
        class_weight="balanced",
        C=2.0,
        max_iter=2000,
        random_state=42,
        solver="liblinear",
    )
    nb = ComplementNB(alpha=0.3)
    svc = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=1.0, random_state=42),
        cv=3,
    )
    ensemble = VotingClassifier(
        estimators=[("lr", logreg), ("nb", nb), ("svc", svc)],
        voting="soft",
        weights=[2, 1, 2],
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", ensemble),
        ]
    )


def best_f1_threshold(y_true, y_proba) -> float:
    # Pick the probability threshold that maximizes F1 on validation.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def main() -> None:
    train_csv = resolve_train_csv()

    df = pd.read_csv(train_csv)
    keyword = df["keyword"] if "keyword" in df.columns else ""
    df["model_input"] = [
        build_input(t, k)
        for t, k in zip(df["text"], keyword if hasattr(keyword, "__iter__") else [""] * len(df))
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        df["model_input"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    val_pred = (val_proba >= threshold).astype(int)

    f1 = f1_score(y_val, val_pred)
    f1_default = f1_score(y_val, (val_proba >= 0.5).astype(int))
    print(f"Validation F1 @0.50: {f1_default:.4f}")
    print(f"Validation F1 @{threshold:.3f} (tuned): {f1:.4f}")
    print(classification_report(y_val, val_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "threshold": threshold}, f)
    print(f"Model saved to {MODEL_PATH} (threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
