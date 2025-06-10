import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def get_diverse_samples(X_class, n_samples, method='kmeans'):
    if len(X_class) <= n_samples:
        return X_class.index.tolist()

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(X_class)
        distances = pairwise_distances(X_class, kmeans.cluster_centers_).min(axis=1)
        selected_indices = distances.argsort()[:n_samples]
        return X_class.iloc[selected_indices].index.tolist()
    else:
        raise NotImplementedError("Only 'kmeans' method is implemented.")

def create_representative_train_set(df, target_col, train_size=0.7, max_per_class=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    selected_indices = []
    for cls in y.unique():
        cls_indices = y[y == cls].index
        X_cls = X_scaled_df.loc[cls_indices]
        n_cls_samples = int(train_size * len(cls_indices))
        if max_per_class is not None:
            n_cls_samples = min(n_cls_samples, max_per_class)

        diverse_ids = get_diverse_samples(X_cls, n_cls_samples)
        selected_indices.extend(diverse_ids)

    train_df = df.loc[selected_indices]
    test_df = df.drop(index=selected_indices)

    return train_df, test_df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_data_path = project_root / "data" / "raw" / "fetal_health.csv"

    df = pd.read_csv(raw_data_path)
    target_column = "fetal_health"
    train_set, test_set = create_representative_train_set(df, target_column)

    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
