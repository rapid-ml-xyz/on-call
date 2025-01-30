import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch_frame import stype, TaskType
from src.on_call.model_pipeline import ModelPipeline, PipelineMetadata, ReferenceData


@dataclass
class MockModel:
    model: any
    task_params: dict

    def create_metadata(self, X):
        pass


@pytest.fixture
def sample_classification_data():
    df = pd.DataFrame({
        'id': range(100),
        'timestamp': pd.date_range(start='2024-01-01', periods=100),
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return df


@pytest.fixture
def sample_regression_data():
    df = pd.DataFrame({
        'id': range(100),
        'timestamp': pd.date_range(start='2024-01-01', periods=100),
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100)
    })
    return df


@pytest.fixture
def classification_pipeline(sample_classification_data):
    model = MockModel(
        model=LogisticRegression(),
        task_params={
            'identifier_cols': ['id', 'timestamp'],
            'target_col': 'target',
            'task_type': TaskType.BINARY_CLASSIFICATION
        }
    )

    col_to_stype = {
        'id': stype.categorical,
        'timestamp': stype.timestamp,
        'feature1': stype.numerical,
        'feature2': stype.categorical,
        'target': stype.categorical
    }

    pipeline = ModelPipeline(model=model, col_to_stype=col_to_stype)
    pipeline.fit(sample_classification_data)
    return pipeline


@pytest.fixture
def regression_pipeline(sample_regression_data):
    model = MockModel(
        model=LinearRegression(),
        task_params={
            'identifier_cols': ['id', 'timestamp'],
            'target_col': 'target',
            'task_type': TaskType.REGRESSION
        }
    )

    col_to_stype = {
        'id': stype.categorical,
        'timestamp': stype.timestamp,
        'feature1': stype.numerical,
        'feature2': stype.categorical,
        'target': stype.numerical
    }

    pipeline = ModelPipeline(model=model, col_to_stype=col_to_stype)
    pipeline.fit(sample_regression_data)
    return pipeline


def test_pipeline_initialization(classification_pipeline):
    assert classification_pipeline.pipeline is not None
    assert isinstance(classification_pipeline.metadata, PipelineMetadata)
    assert classification_pipeline.get_feature_columns() == ['feature1', 'feature2']


def test_feature_column_getters(classification_pipeline):
    assert classification_pipeline.get_numerical_feature_columns() == ['feature1']
    assert classification_pipeline.get_categorical_feature_columns() == ['feature2']
    assert classification_pipeline.get_timestamp_feature_columns() == []


def test_prepare_data(classification_pipeline, sample_classification_data):
    X, y = classification_pipeline.prepare_data(sample_classification_data)
    assert set(X.columns) == {'feature1', 'feature2'}
    assert len(y) == len(sample_classification_data)


def test_predict(classification_pipeline, sample_classification_data):
    predictions = classification_pipeline.predict(sample_classification_data)
    assert len(predictions) == len(sample_classification_data)
    assert all(isinstance(pred, (int, np.int64)) for pred in predictions)


def test_get_diff_classification(classification_pipeline, sample_classification_data):
    predictions = classification_pipeline.predict(sample_classification_data)
    diff = classification_pipeline.get_diff(sample_classification_data, predictions)
    assert len(diff) == len(sample_classification_data)
    assert all(isinstance(d, bool) for d in diff)


def test_get_diff_regression(regression_pipeline, sample_regression_data):
    predictions = regression_pipeline.predict(sample_regression_data)
    diff = regression_pipeline.get_diff(sample_regression_data, predictions)
    assert len(diff) == len(sample_regression_data)
    assert all(isinstance(d, float) for d in diff)


def test_predict_and_append_to_df(classification_pipeline, sample_classification_data):
    result_df = classification_pipeline.predict_and_append_to_df(sample_classification_data)
    assert 'prediction' in result_df.columns
    assert 'error' in result_df.columns
    assert len(result_df) == len(sample_classification_data)


def test_enrich_with_metrics_classification(classification_pipeline, sample_classification_data):
    classification_pipeline.enrich_with_metrics(sample_classification_data)
    metrics = classification_pipeline.metadata.baseline_metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics


def test_enrich_with_metrics_regression(regression_pipeline, sample_regression_data):
    regression_pipeline.enrich_with_metrics(sample_regression_data)
    metrics = regression_pipeline.metadata.baseline_metrics
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_save_and_load(classification_pipeline, tmp_path):
    save_path = tmp_path / "pipeline.joblib"
    classification_pipeline.save(str(save_path))

    loaded_pipeline = ModelPipeline.load(str(save_path))

    assert loaded_pipeline.pipeline is not None
    assert isinstance(loaded_pipeline.metadata, PipelineMetadata)
    assert loaded_pipeline.col_to_stype == classification_pipeline.col_to_stype


def test_enrich_with_ref_data(classification_pipeline, sample_classification_data):
    train_df = sample_classification_data.iloc[:60]
    val_df = sample_classification_data.iloc[60:80]
    test_df = sample_classification_data.iloc[80:]

    classification_pipeline.enrich_with_ref_data(train_df, val_df, test_df)

    assert classification_pipeline.ref_data is not None
    assert len(classification_pipeline.ref_data.train_df) == 60
    assert len(classification_pipeline.ref_data.val_df) == 20
    assert len(classification_pipeline.ref_data.test_df) == 20


def test_invalid_ref_data_modification(classification_pipeline, sample_classification_data):
    train_df = val_df = test_df = sample_classification_data
    classification_pipeline.enrich_with_ref_data(train_df, val_df, test_df)
    with pytest.raises(ValueError):
        classification_pipeline.enrich_with_ref_data(train_df, val_df, test_df)


def test_error_handling_missing_target(classification_pipeline, sample_classification_data):
    df_no_target = sample_classification_data.drop('target', axis=1)
    with pytest.raises(ValueError):
        classification_pipeline.get_diff(df_no_target, np.zeros(len(df_no_target)))


def test_reference_data_creation():
    train_df = pd.DataFrame({'A': [1, 2, 3]})
    val_df = pd.DataFrame({'A': [4, 5]})
    test_df = pd.DataFrame({'A': [6, 7]})

    ref_data = ReferenceData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    assert ref_data.train_df.equals(train_df)
    assert ref_data.val_df.equals(val_df)
    assert ref_data.test_df.equals(test_df)


def test_reference_data_optional_fields():
    ref_data = ReferenceData(
        train_df=None,
        val_df=None,
        test_df=None
    )

    assert ref_data.train_df is None
    assert ref_data.val_df is None
    assert ref_data.test_df is None


def test_reference_data_mixed_optional():
    train_df = pd.DataFrame({'A': [1, 2, 3]})

    ref_data = ReferenceData(
        train_df=train_df,
        val_df=None,
        test_df=None
    )

    assert ref_data.train_df.equals(train_df)
    assert ref_data.val_df is None
    assert ref_data.test_df is None


def test_reference_data_immutability():
    original_df = pd.DataFrame({'A': [1, 2, 3]})
    ref_data = ReferenceData(
        train_df=original_df.copy(),
        val_df=None,
        test_df=None
    )
    before_modification = ref_data.train_df.copy()
    original_df.loc[0, 'A'] = 999
    pd.testing.assert_frame_equal(ref_data.train_df, before_modification)
    assert not ref_data.train_df.equals(original_df)


def test_reference_data_with_different_schemas():
    train_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    val_df = pd.DataFrame({'A': [5], 'C': [6]})
    test_df = pd.DataFrame({'B': [7, 8], 'C': [9, 10]})

    ref_data = ReferenceData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    assert set(ref_data.train_df.columns) == {'A', 'B'}
    assert set(ref_data.val_df.columns) == {'A', 'C'}
    assert set(ref_data.test_df.columns) == {'B', 'C'}


def test_prediction_columns_in_reference_data(classification_pipeline, sample_classification_data):
    train_df = sample_classification_data.iloc[:60]
    val_df = sample_classification_data.iloc[60:80]
    test_df = classification_pipeline.predict_and_append_to_df(sample_classification_data.iloc[80:])

    classification_pipeline.enrich_with_ref_data(train_df, val_df, test_df)

    assert 'prediction' in classification_pipeline.ref_data.test_df.columns
    assert 'error' in classification_pipeline.ref_data.test_df.columns
    assert len(classification_pipeline.ref_data.test_df) == 20


def test_no_reference_data_in_saved_pipeline(classification_pipeline, sample_classification_data, tmp_path):
    train_df = sample_classification_data.iloc[:60]
    val_df = sample_classification_data.iloc[60:80]
    test_df = sample_classification_data.iloc[80:]

    classification_pipeline.enrich_with_ref_data(train_df, val_df, test_df)

    save_path = tmp_path / "pipeline.joblib"
    classification_pipeline.save(str(save_path))

    loaded_pipeline = ModelPipeline.load(str(save_path))
    assert loaded_pipeline.ref_data is None
