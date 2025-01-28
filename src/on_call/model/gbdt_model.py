from data.hm.loader import prepare_data
from torch_frame import TaskType
from torch_frame.gbdt import LightGBM, XGBoost

NUM_TRIALS = 10


def get_model(dataset: str, task: str, booster: str = 'lgbm', subsample: int = 0, init_db: bool = False,
              generate_feats: bool = False, drop_cols: list = None) -> tuple:
    task_obj, task_params, train_dset, val_tf, test_tf, val_task_df = prepare_data(
        dataset, task, subsample, init_db, generate_feats, drop_cols
    )

    booster_cls = LightGBM if booster == 'lgbm' else XGBoost
    if task_params['task_type'] == TaskType.BINARY_CLASSIFICATION:
        gbdt = booster_cls(task_params['task_type'], num_classes=2, metric=task_params['tune_metric'])
    elif task_params['task_type'] == TaskType.REGRESSION:
        gbdt = booster_cls(task_params['task_type'], metric=task_params['tune_metric'])
    else:
        raise ValueError(f"{task_params['task_type']} not supported")

    gbdt.tune(tf_train=train_dset.tensor_frame, tf_val=val_tf, num_trials=NUM_TRIALS)

    val_pred = gbdt.predict(tf_test=val_tf).numpy()
    test_pred = gbdt.predict(tf_test=test_tf).numpy()

    val_table = task_obj.get_table("val")
    val_table.df = val_task_df
    val_metrics = task_obj.evaluate(val_pred, val_table)

    return gbdt, val_metrics, test_pred
