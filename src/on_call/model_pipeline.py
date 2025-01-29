import os
import pandas as pd
import sklearn
from dataclasses import dataclass
from typing import Dict, List, Optional
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from packaging import version
from torch_frame import stype, TaskType
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    mean_squared_error, mean_absolute_error, r2_score

if version.parse(sklearn.__version__) < version.parse('1.2'):
    ohe_params = {"sparse": False}
else:
    ohe_params = {"sparse_output": False}


@dataclass
class PipelineMetadata:
    numerical_columns: List[str]
    categorical_columns: List[str]
    timestamp_columns: List[str]
    feature_columns: List[str]
    column_types: Dict[str, stype]


class TypeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str)


class ModelPipeline:
    def __init__(
        self,
        model,
        col_to_stype: Dict[str, stype],
        df: pd.DataFrame,
        skip_build: bool = False
    ):
        self._model = model
        self._col_to_stype = col_to_stype
        self._pipeline = None
        self._metadata = self._create_pipeline_metadata(df)
        if not skip_build:
            self._build_pipeline()

    @property
    def model(self):
        return self._model

    @property
    def col_to_stype(self) -> Dict[str, stype]:
        return self._col_to_stype

    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline

    @property
    def metadata(self) -> PipelineMetadata:
        return self._metadata

    def _create_pipeline_metadata(self, df: pd.DataFrame) -> PipelineMetadata:
        exclude_cols = (self._model.task_params['identifier_cols'] + [self._model.task_params['target_col']])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        numerical_cols = [
            col for col, st in self._col_to_stype.items() 
            if st == stype.numerical and col in feature_cols
        ]
        categorical_cols = [
            col for col, st in self._col_to_stype.items() 
            if st == stype.categorical and col in feature_cols
        ]
        timestamp_cols = [
            col for col, st in self._col_to_stype.items() 
            if st == stype.timestamp and col in feature_cols
        ]

        return PipelineMetadata(
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            timestamp_columns=timestamp_cols,
            feature_columns=feature_cols,
            column_types=self._col_to_stype
        )

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[self._metadata.feature_columns]
        y = df[self._model.task_params['target_col']] if self._model.task_params['target_col'] in df else None
        return X, y

    def _build_pipeline(self):
        transformers = []

        if self._metadata.numerical_columns:
            num_pipe = Pipeline([
                ('num_imputer', SimpleImputer(strategy='median')),
                ('num_scaler', StandardScaler())
            ])
            transformers.append(('num_pipe', num_pipe, self._metadata.numerical_columns))

        cat_cols = self._metadata.categorical_columns + self._metadata.timestamp_columns
        if cat_cols:
            cat_pipe = Pipeline([
                ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('type_converter', TypeConverter()),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', **ohe_params))
            ])
            transformers.append(('cat_pipe', cat_pipe, cat_cols))

        if not transformers:
            raise ValueError("No valid columns found for preprocessing")

        preprocessor = ColumnTransformer(transformers)

        self._pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', self._model.model)
        ])

    def fit(self, df: pd.DataFrame):
        X, y = self.prepare_data(df)
        if y is None:
            raise ValueError("Target column not found in dataframe")
        self._pipeline.fit(X, y)
        self._model.create_metadata(X)
        return self

    def predict(self, df: pd.DataFrame):
        X, _ = self.prepare_data(df)
        return self._pipeline.predict(X)
        
    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dump({
            'pipeline': self._pipeline,
            'metadata': self._metadata,
            'col_to_stype': self._col_to_stype,
            'model': self._model,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelPipeline':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
            
        try:
            saved_data = load(filepath)
            pipeline = cls(
                model=saved_data['model'],
                col_to_stype=saved_data['col_to_stype'],
                df=pd.DataFrame(),
                skip_build=True
            )
            pipeline._pipeline = saved_data['pipeline']
            pipeline._metadata = saved_data['metadata']
            return pipeline
            
        except Exception as e:
            raise ValueError(f"Failed to load pipeline: {str(e)}")

    def get_metrics(self, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        X, y_true = self.prepare_data(df)
        if y_true is None:
            raise ValueError("Target column not found in dataframe")

        y_pred = self.predict(df)
        task_type = self._model.task_params['task_type']

        if metrics is None:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            elif task_type == TaskType.REGRESSION:
                metrics = ['mse', 'mae', 'r2']

        results = {}
        for metric in metrics:
            try:
                if metric == 'accuracy':
                    results[metric] = accuracy_score(y_true, y_pred)
                elif metric == 'precision':
                    results[metric] = precision_score(y_true, y_pred, average='weighted')
                elif metric == 'recall':
                    results[metric] = recall_score(y_true, y_pred, average='weighted')
                elif metric == 'f1':
                    results[metric] = f1_score(y_true, y_pred, average='weighted')
                elif metric == 'mse':
                    results[metric] = mean_squared_error(y_true, y_pred)
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(y_true, y_pred)
                elif metric == 'r2':
                    results[metric] = r2_score(y_true, y_pred)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            except Exception as e:
                results[metric] = f"Error calculating {metric}: {str(e)}"

        return results
