import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

splits = {'train': 'data/train-00000-of-00001.parquet',
          'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet(
    "hf://datasets/aRWA787/ag_news_dataset/" + splits["train"])

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000
        )
    )
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(
    classification_report(
        y_test,
        preds
    )
)

label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}
joblib.dump(label_map, "training/label_map.pkl")
joblib.dump(
    model,
    "training/model.pkl"
)
