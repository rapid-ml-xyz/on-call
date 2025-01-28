import duckdb

from .inferred_stypes import task_to_stypes
from .utils import db_setup, render_jinja_sql
from relbench.tasks import get_task
from torch_frame import TaskType, stype
from torch_frame.data import Dataset
from torch_frame.typing import Metric

DATASET_TO_DB = {'rel-hm': 'data/hm/hm.db'}

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


def get_matching_rows(feats_df, labels_df, identifier_cols, n_rows=1000):
    feat_str_df = feats_df[identifier_cols].astype(str)
    label_str_df = labels_df[identifier_cols].astype(str)

    feat_keys = feat_str_df.agg('-'.join, axis=1)
    label_keys = label_str_df.agg('-'.join, axis=1)

    common_keys = set(feat_keys) & set(label_keys)
    if len(common_keys) < n_rows:
        raise ValueError(f'Only found {len(common_keys)} matching rows, needed {n_rows}')

    selected_keys = list(common_keys)[:n_rows]
    matched_feats = feats_df[feat_keys.isin(selected_keys)]
    matched_labels = labels_df[label_keys.isin(selected_keys)]

    matched_feats['_key'] = feat_keys[matched_feats.index]
    matched_labels['_key'] = label_keys[matched_labels.index]

    matched_feats = matched_feats.sort_values('_key').drop('_key', axis=1)
    matched_labels = matched_labels.sort_values('_key').drop('_key', axis=1)

    return matched_feats.reset_index(drop=True), matched_labels.reset_index(drop=True)


def prepare_data(dataset: str, task: str, subsample: int = 0, init_db: bool = False,
                 generate_feats: bool = False, drop_cols: list = None) -> tuple:
    if init_db:
        db_setup(dataset, DATASET_TO_DB[dataset])

    drop_cols = drop_cols or []
    full_task_name = f'{dataset}-{task}'
    task_params = TASK_PARAMS[full_task_name]

    task_obj = get_task(dataset, task, download=True)
    val_task_df = task_obj.get_table("val").df
    test_task_df = task_obj.get_table("test").df

    conn = duckdb.connect(DATASET_TO_DB[dataset])

    if generate_feats:
        with open('build/hm/feats.sql') as f:
            template = f.read()
        for s in ['train', 'val', 'test']:
            query = render_jinja_sql(template, dict(set=s, subsample=subsample))
            conn.sql(query)

    train_df = conn.sql(f'select * from {task_params["table_prefix"]}_train_feats').df()
    val_df = conn.sql(f'select * from {task_params["table_prefix"]}_val_feats').df()
    test_df = conn.sql(f'select * from {task_params["table_prefix"]}_test_feats').df()
    conn.close()

    val_df, val_task_df = get_matching_rows(val_df, val_task_df, task_params['identifier_cols'])
    test_df, test_task_df = get_matching_rows(test_df, test_task_df, task_params['identifier_cols'])

    col_to_stype = task_to_stypes[full_task_name]
    drop_cols = task_params['identifier_cols'] + drop_cols

    train_df = train_df.drop(drop_cols, axis=1)
    val_df = val_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)

    for col in drop_cols:
        if col in col_to_stype:
            del col_to_stype[col]

    if subsample > 0 and not generate_feats:
        train_df = train_df.head(subsample)

    for k, v in col_to_stype.items():
        if v == stype.text_embedded:
            raise NotImplementedError('Embeddings for text columns not supported for speed considerations.')

    train_dset = Dataset(train_df, col_to_stype=col_to_stype, target_col=task_params['target_col']).materialize()
    val_tf = train_dset.convert_to_tensor_frame(val_df)
    test_tf = train_dset.convert_to_tensor_frame(test_df)

    return task_obj, task_params, train_dset, val_tf, test_tf, val_task_df
