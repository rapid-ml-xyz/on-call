from abc import abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

from ..constants import ModelTask
from ..utils.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


class BaseModel():
    """
    Base class for all models.
    params:
        df: input dataframe
        cat_features: list of categorical features
        num_features: list of numerical features
        target: target column
        test_size: test size for train/test split
        random_state: random state for train/test split
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_features: list[str],
        num_features: list[str],
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        model_task: ModelTask = ModelTask.CLASSIFICATION,
        model_path: str = None,
        model_params: dict = None,
    ):
        self.df = df
        self.cat_features = [feat.lower() for feat in cat_features]
        self.num_features = [feat.lower() for feat in num_features]
        self.target = target.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.model_task: ModelTask = model_task
        self.model_path: str = model_path
        self.model_params: dict = model_params
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.eval_metrics: dict[str, float] = None
        self.preprocess_data()

    def preprocess_data(self):
        self.process_features()
        X = self.df[self.cat_features + self.num_features]
        y = self.df[self.target].astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self._align_categories()

    def _align_categories(self):
        for col in self.cat_features:
            self.X_train[col] = (
                self.X_train[col]
                .cat.add_categories(
                    [
                        x
                        for x in self.X_test[col].cat.categories
                        if x not in self.X_train[col].cat.categories
                    ]
                )
                .cat.set_categories(self.X_train[col].cat.categories)
            )
            self.X_test[col] = self.X_test[col].cat.set_categories(
                self.X_train[col].cat.categories
            )

    def process_features(self):
        for column in self.cat_features + self.num_features:
            if column in self.cat_features:
                self.df[column] = self.df[column].astype("category")
            else:
                self.df[column] = pd.to_numeric(self.df[column], errors="coerce")

    @abstractmethod
    def train_model(self):
        pass

    def evaluate_model(self, pred_y, true_y):
        if self.model_task == ModelTask.CLASSIFICATION:
            self.eval_metrics = calculate_classification_metrics(pred_y, true_y)
        elif self.model_task == ModelTask.REGRESSION:
            self.eval_metrics = calculate_regression_metrics(pred_y, true_y)

    def save_model(self):
        if self.model_path:
            self.model.save_model(self.model_path)

    @abstractmethod
    def load_model(self):
        pass
