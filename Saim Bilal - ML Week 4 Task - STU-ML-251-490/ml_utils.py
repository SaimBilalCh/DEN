import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ucimlrepo import fetch_ucirepo


def load_data():
    """Fetch and preprocess the dataset."""
    breast_cancer = fetch_ucirepo(id=17)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, X.columns.tolist()


def feature_selection(X_train, y_train, X_test, k=10):
    """Select top k features using ANOVA F-test."""
    selector = SelectKBest(f_classif, k=k)
    X_train_fs = selector.fit_transform(X_train, y_train)
    X_test_fs = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    return X_train_fs, X_test_fs, selected_features


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_and_evaluate(k=10):
    """Train models with/without FS and return results + trained models."""
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #With Feature Selection
    X_train_fs, X_test_fs, selected_features = feature_selection(X_train, y_train, X_test, k)

    models = {
        "Logistic Regression (No FS)": LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train),
        "Random Forest (No FS)": RandomForestClassifier(random_state=42).fit(X_train, y_train),
        "Logistic Regression (With FS)": LogisticRegression(random_state=42, max_iter=1000).fit(X_train_fs, y_train),
        "Random Forest (With FS)": RandomForestClassifier(random_state=42).fit(X_train_fs, y_train),
    }

    results = {}
    for name, model in models.items():
        if "With FS" in name:
            results[name] = evaluate_model(model, X_test_fs, y_test)
        else:
            results[name] = evaluate_model(model, X_test, y_test)

    #Save results + models
    with open("ml_results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)

    return results, selected_features
