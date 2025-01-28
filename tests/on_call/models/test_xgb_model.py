import pytest
import pandas as pd
import numpy as np
from src.on_call.model.xgb_model import XGBModel
from src.on_call.constants import ModelTask


def setup_telco_churn_data():
    """Setup test data from telco churn dataset"""
    df0 = pd.read_csv("data/telco_churn_data.csv", index_col=False)
    df0.columns = [x.lower() for x in df0.columns]

    # Define feature columns
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
    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    target = "churn"

    # Convert target to numeric
    df0[target] = df0[target].apply(lambda x: 1 if x == "Yes" else 0)

    # Convert categorical columns
    df0[cat_cols] = df0[cat_cols].astype("category")

    # Convert numeric columns
    for col in num_cols:
        df0[col] = pd.to_numeric(df0[col], errors="coerce")

    return df0, cat_cols, num_cols, target


@pytest.fixture
def model_params():
    """Fixture for model parameters"""
    return {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}


@pytest.fixture
def test_data():
    """Fixture for test data"""
    df, cat_cols, num_cols, target = setup_telco_churn_data()
    return {
        "df": df,
        "cat_features": cat_cols,
        "num_features": num_cols,
        "target": target,
    }


def test_xgb_model_initialization(test_data, model_params):
    """Test XGBoost model initialization"""
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )

    assert model is not None
    assert model.model_task == ModelTask.CLASSIFICATION
    assert model.cat_features == [x.lower() for x in test_data["cat_features"]]
    assert model.num_features == [x.lower() for x in test_data["num_features"]]
    assert model.target == test_data["target"].lower()


def test_data_preprocessing(test_data, model_params):
    """Test data preprocessing"""
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )

    # Check if train-test split was performed
    assert model.X_train is not None
    assert model.X_test is not None
    assert model.y_train is not None
    assert model.y_test is not None

    # Check if features were processed correctly
    for col in model.cat_features:
        assert model.X_train[col].dtype.name == "category"

    for col in model.num_features:
        assert np.issubdtype(model.X_train[col].dtype, np.number)


def test_model_training(test_data, model_params):
    """Test model training"""
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )

    model.train_model()

    # Check if model was trained
    assert model.model is not None
    assert hasattr(model.model, "predict")

    # Check if predictions can be made
    predictions = model.model.predict(model.X_test)
    assert len(predictions) == len(model.y_test)

    # Check if evaluation metrics were calculated
    assert model.eval_metrics is not None
    assert "accuracy_score" in model.eval_metrics
    assert "precision_score" in model.eval_metrics
    assert "recall_score" in model.eval_metrics
    assert "f1_score" in model.eval_metrics


def test_model_save_load(test_data, model_params, tmp_path):
    """Test model saving and loading"""
    model_path = tmp_path / "test_model.json"

    # Create and train model
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
        model_path=str(model_path),
    )

    model.train_model(save_model=True)
    original_predictions = model.model.predict(model.X_test)

    # Create new model instance and load saved model
    loaded_model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
        model_path=str(model_path),
    )

    loaded_model.load_model()
    loaded_predictions = loaded_model.model.predict(model.X_test)

    # Check if predictions match
    assert np.array_equal(original_predictions, loaded_predictions)


def test_model_evaluation(test_data, model_params):
    """Test model evaluation metrics"""
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )

    model.train_model()

    # Check evaluation metrics
    assert isinstance(model.eval_metrics, dict)
    assert all(
        metric in model.eval_metrics
        for metric in ["accuracy_score", "precision_score", "recall_score", "f1_score"]
    )
    assert all(0 <= model.eval_metrics[metric] <= 1 for metric in model.eval_metrics)


def test_error_handling(test_data, model_params):
    """Test error handling for invalid inputs"""

    # Test with invalid categorical feature
    with pytest.raises(KeyError):
        XGBModel(
            df=test_data["df"],
            cat_features=test_data["cat_features"] + ["invalid_feature"],
            num_features=test_data["num_features"],
            target=test_data["target"],
            model_params=model_params,
        )

    # Test with invalid numerical feature
    with pytest.raises(KeyError):
        XGBModel(
            df=test_data["df"],
            cat_features=test_data["cat_features"],
            num_features=test_data["num_features"] + ["invalid_feature"],
            target=test_data["target"],
            model_params=model_params,
        )

    # Test with invalid target
    with pytest.raises(KeyError):
        XGBModel(
            df=test_data["df"],
            cat_features=test_data["cat_features"],
            num_features=test_data["num_features"],
            target="invalid_target",
            model_params=model_params,
        )


def test_prediction_shape_and_values(test_data, model_params):
    """Test prediction shapes and value ranges"""
    model = XGBModel(
        df=test_data["df"],
        cat_features=test_data["cat_features"],
        num_features=test_data["num_features"],
        target=test_data["target"],
        model_params=model_params,
    )

    model.train_model()
    predictions = model.model.predict(model.X_test)

    # Check prediction shape
    assert len(predictions) == len(model.y_test)

    # Check prediction values (should be 0 or 1 for binary classification)
    assert set(predictions).issubset({0, 1})
