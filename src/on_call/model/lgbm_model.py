from dataclasses import dataclass
from typing import Dict, Optional
from torch_frame import TaskType
from lightgbm import LGBMClassifier, LGBMRegressor

SEED_VALUE = 42


@dataclass
class ModelMetadata:
    task_type: TaskType
    target_column: str
    identifier_columns: list
    model_parameters: Dict
    num_features: int
    num_train_samples: int


class LGBMModel:
    def __init__(
        self,
        task_params: Dict,
        col_to_stype: Dict,
        model_params: Optional[Dict] = None
    ):
        self._task_params = task_params
        self._col_to_stype = col_to_stype
        self._model_params = model_params or {}
        self._model = None
        self._metadata = None
        self._initialize_model()

    @property
    def task_params(self) -> Dict:
        return self._task_params

    @property
    def col_to_stype(self) -> Dict:
        return self._col_to_stype

    @property
    def model_params(self) -> Dict:
        return self._model_params

    @property
    def model(self):
        return self._model

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    def _initialize_model(self):
        if self._task_params['task_type'] == TaskType.BINARY_CLASSIFICATION:
            self._model = LGBMClassifier(random_state=SEED_VALUE, **self._model_params)
        elif self._task_params['task_type'] == TaskType.REGRESSION:
            self._model = LGBMRegressor(random_state=SEED_VALUE, **self._model_params)
        else:
            raise ValueError(f"{self._task_params['task_type']} not supported")

    def create_metadata(self, X_train):
        self._metadata = ModelMetadata(
            task_type=self._task_params['task_type'],
            target_column=self._task_params['target_col'],
            identifier_columns=self._task_params['identifier_cols'],
            model_parameters=self._model_params,
            num_features=X_train.shape[1],
            num_train_samples=X_train.shape[0]
        )
        return self._metadata
