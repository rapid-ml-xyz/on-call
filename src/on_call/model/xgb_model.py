import xgboost as xgb

from .base_model import BaseModel


class XGBModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_model(self, save_model: bool = False):
        self.model = xgb.XGBClassifier(**self.model_params, tree_method="hist", early_stopping_rounds=2, enable_categorical=True)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)]
        )
        self.pred = self.model.predict(self.X_test)
        self.evaluate_model(self.y_test, self.pred)
        if save_model:
            self.model.save_model(self.model_path)

    def load_model(self):
        if self.model_path:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
