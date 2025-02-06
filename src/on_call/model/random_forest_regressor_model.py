from sklearn import ensemble
import pandas as pd
from typing import List, Tuple


class RandomForestRegressor:
    def __init__(
        self,
        target: str,
        prediction_col: str = 'prediction',
        n_estimators: int = 50,
        random_state: int = 0
    ):
        self.target = target
        self.prediction_col = prediction_col
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.model = ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.reference_data: pd.DataFrame = pd.DataFrame()
        self.current_data: pd.DataFrame = pd.DataFrame()

    def set_features(
        self,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def split_data(
        self,
        data: pd.DataFrame,
        reference_end_date: str,
        current_end_date: str,
        start_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be datetime")
        if start_date:
            self.reference_data = data.loc[start_date:reference_end_date].copy()
        else:
            self.reference_data = data.loc[:reference_end_date].copy()
        self.current_data = data.loc[reference_end_date:current_end_date].copy()
        return self.reference_data, self.current_data

    def train(self) -> None:
        if not (self.numerical_features or self.categorical_features):
            raise ValueError("Features not set")
        if self.reference_data is None:
            raise ValueError("Data not split")

        features = self.numerical_features + self.categorical_features
        self.model.fit(
            self.reference_data[features],
            self.reference_data[self.target]
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        features = self.numerical_features + self.categorical_features
        return pd.Series(
            self.model.predict(data[features]),
            index=data.index,
            name=self.prediction_col
        )

    def generate_predictions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.reference_data[self.prediction_col] = self.predict(self.reference_data)
        self.current_data[self.prediction_col] = self.predict(self.current_data)
        return self.reference_data, self.current_data
