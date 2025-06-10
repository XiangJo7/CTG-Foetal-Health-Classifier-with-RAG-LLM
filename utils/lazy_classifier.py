from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def main():
    # == Project Root ==
    project_root = Path(__file__).resolve().parents[1]

    # === Load Data ===
    train_data_path = project_root / "data" / "processed" / "train.csv"
    df_train = pd.read_csv(train_data_path)

    X = df_train.drop("fetal_health", axis=1)
    y = df_train["fetal_health"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize LazyClassifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True)

    # Fit and evaluate
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Show top 5 models
    print("Top 5 models based on accuracy:")
    top_models = models.head(5)
    print(top_models)

if __name__ == "__main__":
    main()
