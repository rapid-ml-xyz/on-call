from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from packaging import version
from torch_frame import stype
import pandas as pd
import sklearn

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
        df: pd.DataFrame
    ):
        self._model = model
        self._col_to_stype = col_to_stype
        self._pipeline = None
        self._metadata = self._create_pipeline_metadata(df)
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
        exclude_cols = (self._model.task_params['identifier_cols'] +
                       [self._model.task_params['target_col']])
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

        if self.metadata.numerical_columns:
            num_pipe = Pipeline([
                ('num_imputer', SimpleImputer(strategy='median')),
                ('num_scaler', StandardScaler())
            ])
            transformers.append(('num_pipe', num_pipe, self.metadata.numerical_columns))

        cat_cols = self.metadata.categorical_columns + self.metadata.timestamp_columns
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
