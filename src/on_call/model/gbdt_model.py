from data.hm.inferred_stypes import task_to_stypes
from data.hm.loader import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from packaging import version
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from torch_frame import stype, TaskType

NUM_TRIALS = 10
SEED_VALUE = 42

if version.parse(sklearn.__version__) < version.parse('1.2'):
    ohe_params = {"sparse": False}
else:
    ohe_params = {"sparse_output": False}


class TypeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str)


def create_classification_pipeline(df, col_to_stype):
    num_cols = [col for col, st in col_to_stype.items() if st == stype.numerical and col in df.columns]
    cat_cols = [col for col, st in col_to_stype.items() if st == stype.categorical and col in df.columns]
    timestamp_cols = [col for col, st in col_to_stype.items() if st == stype.timestamp and col in df.columns]

    cat_cols = cat_cols + timestamp_cols
    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('num_scaler', StandardScaler())
        ])
        transformers.append(('num_pipe', num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('type_converter', TypeConverter()),  # Convert all values to strings
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', **ohe_params))
        ])
        transformers.append(('cat_pipe', cat_pipe, cat_cols))

    return ColumnTransformer(transformers)


def get_model(
        dataset: str,
        task: str,
        subsample: int = 0,
        init_db: bool = False,
        generate_feats: bool = False
) -> Pipeline:
    task_obj, task_params, train_df, val_df, test_df = prepare_data(
        dataset, task, subsample, init_db, generate_feats
    )
    full_task_name = f'{dataset}-{task}'
    col_to_stype = task_to_stypes[full_task_name]

    exclude_cols = task_params['identifier_cols'] + [task_params['target_col']]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[task_params['target_col']]

    preprocessor = create_classification_pipeline(X_train, col_to_stype)

    if task_params['task_type'] == TaskType.BINARY_CLASSIFICATION:
        model = LGBMClassifier(random_state=SEED_VALUE)
    elif task_params['task_type'] == TaskType.REGRESSION:
        model = LGBMRegressor(random_state=SEED_VALUE)
    else:
        raise ValueError(f"{task_params['task_type']} not supported")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    X_val = val_df[feature_cols]
    val_pred = pipeline.predict(X_val)
    print(val_pred)

    X_test = test_df[feature_cols]
    test_pred = pipeline.predict(X_test)
    print(test_pred)

    return pipeline
