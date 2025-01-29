import pandas as pd
import numpy as np
from enum import Enum
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from src.on_call.model.xgb_model import XGBModel


class DataType(Enum):
    REGULAR = "regular"
    NOISY = "noisy"


class ModelType(Enum):
    REGULAR = "regular"
    UNDER_FITTING = "under-fitting"


def get_telco_churn_data(data_type: DataType = DataType.REGULAR):
    df0 = pd.read_csv("data/telco_churn_data.csv", index_col=False)
    df0.columns = [x.lower() for x in df0.columns]

    cat_cols = [
        "gender",
        "seniorcitizen",
        "partner",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract",
        "paperlessbilling",
        "paymentmethod",
    ]
    num_cols = [
        "tenure",
        "monthlycharges",
        "totalcharges"
    ]
    target = "churn"

    df0[target] = df0[target].apply(lambda x: 1 if x == "Yes" else 0)
    df0[cat_cols] = df0[cat_cols].astype("category")
    for col in num_cols:
        df0[col] = pd.to_numeric(df0[col], errors="coerce")

    match data_type:
        case DataType.NOISY:
            for col in num_cols:
                noise = np.random.normal(0, df0[col].std() * 0.5, size=len(df0))
                df0[col] = df0[col] + noise

            for col in cat_cols:
                mask = np.random.random(size=len(df0)) < 0.3
                df0.loc[mask, col] = np.random.permutation(df0.loc[mask, col].values)

            flip_mask = np.random.random(size=len(df0)) < 0.15
            df0.loc[flip_mask, target] = 1 - df0.loc[flip_mask, target]

        case DataType.REGULAR:
            pass

        case _:
            raise ValueError(f"Unknown model type: {data_type}")

    return {
        "df": df0,
        "cat_features": cat_cols,
        "num_features": num_cols,
        "target": target,
    }


def get_model_params(model_type: ModelType):
    match model_type:
        case ModelType.UNDER_FITTING:
            return {
                "max_depth": 2,
                "learning_rate": 0.3,
                "n_estimators": 30
            }
        case ModelType.REGULAR:
            return {
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 100
            }
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def get_model(
        data_type: DataType = DataType.REGULAR,
        model_type: ModelType = ModelType.REGULAR
):
    test_data = get_telco_churn_data(data_type)
    model_params = get_model_params(model_type)

    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )
    model.train_model()

    predictions = model.pred
    print(predictions)
    return model


def benchmark_models(regular_model, corrupted_model) -> tuple[dict, dict]:
    """Benchmark performance between regular and corrupted models.

    Args:
        regular_model: The baseline model trained on clean data
        corrupted_model: The model trained on noisy data

    Returns:
        tuple[dict, dict]: Metrics for both models and degradation percentages
    """
    y_true = regular_model.y_test
    y_pred_regular = regular_model.pred
    y_pred_corrupted = corrupted_model.pred

    metrics = {
        'regular': {
            'accuracy': accuracy_score(y_true, y_pred_regular),
            'precision': precision_score(y_true, y_pred_regular),
            'recall': recall_score(y_true, y_pred_regular),
            'f1': f1_score(y_true, y_pred_regular),
            'log_loss': log_loss(y_true, y_pred_regular)
        },
        'corrupted': {
            'accuracy': accuracy_score(y_true, y_pred_corrupted),
            'precision': precision_score(y_true, y_pred_corrupted),
            'recall': recall_score(y_true, y_pred_corrupted),
            'f1': f1_score(y_true, y_pred_corrupted),
            'log_loss': log_loss(y_true, y_pred_corrupted)
        }
    }

    degradation = {
        metric: ((metrics['regular'][metric] - metrics['corrupted'][metric]) / metrics['regular'][metric]) * 100
        for metric in metrics['regular'].keys()
    }

    # Print results in a formatted way
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Regular':<12} {'Corrupted':<12} {'Degradation':<12}")
    print("-" * 50)

    for metric in metrics['regular'].keys():
        print(
            f"{metric:<15} "
            f"{metrics['regular'][metric]:>11.4f} "
            f"{metrics['corrupted'][metric]:>11.4f} "
            f"{degradation[metric]:>11.2f}%"
        )

    return metrics, degradation
