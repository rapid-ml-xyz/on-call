import duckdb
import pandas as pd
from dataclasses import dataclass
from torch_frame import TaskType
from torch_frame.typing import Metric
from on_call.data.hm.utils import db_setup, render_jinja_sql

DATA_FOLDER = 'data/hm'
DATASET_TO_DB = {'rel-hm': f'{DATA_FOLDER}/hm.db'}
TASK_PARAMS = {
    'rel-hm-item-sales': {
        'dir': 'hm/item-sales',
        'target_col': 'sales',
        'table_prefix': 'item_sales',
        'identifier_cols': ['article_id', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-hm-user-churn': {
        'dir': 'hm/user-churn',
        'target_col': 'churn',
        'table_prefix': 'user_churn',
        'identifier_cols': ['customer_id', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    }
}


@dataclass
class TimeConfig:
    """Configuration for temporal splitting of data"""
    year: int
    train_months: range
    val_month: int
    test_month: int

    @classmethod
    def default(cls) -> 'TimeConfig':
        return cls(
            year=2020,
            train_months=range(1, 7),
            val_month=7,
            test_month=8
        )


def split_by_timestamp(df, time_config=None):
    if time_config is None:
        time_config = TimeConfig.default()

    df['date'] = pd.to_datetime(df['timestamp'])
    mask_year = df['date'].dt.year == time_config.year
    df = df[mask_year].copy()

    train_mask = df['date'].dt.month.isin(time_config.train_months)
    val_mask = df['date'].dt.month == time_config.val_month
    test_mask = df['date'].dt.month == time_config.test_month

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_df = train_df.drop('date', axis=1)
    val_df = val_df.drop('date', axis=1)
    test_df = test_df.drop('date', axis=1)

    return train_df, val_df, test_df


def fetch_data(dataset: str, task: str, subsample: int = 0, time_config: TimeConfig = None,
               init_db: bool = False, generate_feats: bool = False) -> tuple:

    if init_db:
        db_setup(dataset, DATASET_TO_DB[dataset])

    full_task_name = f'{dataset}-{task}'
    task_params = TASK_PARAMS[full_task_name]

    conn = duckdb.connect(DATASET_TO_DB[dataset])

    if generate_feats:
        with open(f'{DATA_FOLDER}/feats.sql') as f:
            template = f.read()
        query = render_jinja_sql(template, dict(set='train', subsample=subsample))
        conn.sql(query)

    train_df = conn.sql(f'select * from {task_params["table_prefix"]}_train_feats').df()
    conn.close()

    if subsample > 0 and not generate_feats:
        train_df = train_df.head(subsample)

    train_df, val_df, test_df = split_by_timestamp(train_df, time_config)

    train_df.to_csv(f'{DATA_FOLDER}/train.csv', index=False)
    val_df.to_csv(f'{DATA_FOLDER}/val.csv', index=False)
    test_df.to_csv(f'{DATA_FOLDER}/test.csv', index=False)

    return task_params, train_df, val_df, test_df
