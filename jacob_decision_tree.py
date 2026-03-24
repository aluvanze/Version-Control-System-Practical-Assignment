import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(csv_path, encoding="latin1")

    fraud_target = (data["Order Status"] == "SUSPECTED_FRAUD").astype(int)
    late_target = (data["Delivery Status"] == "Late delivery").astype(int)

    drop_columns = [
        "Order Status",
        "Delivery Status",
        "Late_delivery_risk",
        "order date (DateOrders)",
        "shipping date (DateOrders)",
    ]
    features = data.drop(columns=drop_columns, errors="ignore").copy()

    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = pd.factorize(features[col].fillna("NA"))[0]
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    return features, fraud_target, late_target


def train_and_score(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_confusion_matrix(cm, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    csv_path = "DataCoSupplyChainDataset.csv"
    x, y_fraud, y_late = load_and_prepare_data(csv_path)

    x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(
        x, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )
    x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(
        x, y_late, test_size=0.2, random_state=42, stratify=y_late
    )

    fraud_result = train_and_score(x_train_f, x_test_f, y_train_f, y_test_f)
    late_result = train_and_score(x_train_l, x_test_l, y_train_l, y_test_l)

    results_df = pd.DataFrame(
        [
            {
                "task": "fraud_detection",
                "accuracy": fraud_result["accuracy"],
                "recall": fraud_result["recall"],
                "f1_score": fraud_result["f1_score"],
            },
            {
                "task": "late_delivery_prediction",
                "accuracy": late_result["accuracy"],
                "recall": late_result["recall"],
                "f1_score": late_result["f1_score"],
            },
        ]
    )
    results_df.to_csv("jacob_decision_tree_results.csv", index=False)

    save_confusion_matrix(
        fraud_result["confusion_matrix"],
        "Decision Tree - Fraud Detection",
        "jacob_fraud_cm.png",
    )
    save_confusion_matrix(
        late_result["confusion_matrix"],
        "Decision Tree - Late Delivery",
        "jacob_late_cm.png",
    )

    print("Saved jacob_decision_tree_results.csv")
    print("Saved jacob_fraud_cm.png")
    print("Saved jacob_late_cm.png")


if __name__ == "__main__":
    main()
