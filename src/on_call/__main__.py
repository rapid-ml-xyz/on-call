from dotenv import load_dotenv
import pandas as pd
import os
import sys

if not __package__:
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)

from on_call.data.hm.inferred_stypes import task_to_stypes
from on_call.model.lgbm_model import LGBMModel
from on_call.model.random_forest_regressor_model import RandomForestRegressor
from on_call.model_pipeline import ModelPipeline
from on_call.orchestrator.engines import LangGraphMessageState
from on_call.workflow.setup import setup_analysis_workflow

SUBSAMPLE = 20_000
DATASET = 'rel-hm'
TASK = 'user-churn'
PIPELINE_PATH = 'saved_pipelines/rel_hm_user_churn_lgbm_20k.joblib'

load_dotenv()


def _generate_pipeline(_task_params, _train_df):
    full_task_name = f'{DATASET}-{TASK}'
    col_to_stype = task_to_stypes[full_task_name]
    model = LGBMModel(_task_params, col_to_stype)

    _pipeline = ModelPipeline(model, col_to_stype)
    _pipeline.fit(_train_df)
    _pipeline.save(PIPELINE_PATH)

    return _pipeline


def build_model(raw_data):
    model = RandomForestRegressor(target='cnt')
    model.set_features(
        numerical_features=['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday'],
        categorical_features=['season', 'holiday', 'workingday']
    )
    model.split_data(
        data=raw_data,
        reference_end_date='2011-01-28 23:00:00',
        current_end_date='2011-02-28 23:00:00'
    )
    model.train()
    model.generate_predictions()
    return model


def run_analysis(model) -> LangGraphMessageState:
    state = {"model": model}
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    result = workflow.run(state)
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/bike_sharing_data.csv", index_col=0, parse_dates=True)
    _model = build_model(df)
    response = run_analysis(_model)
    print(f"Analysis Results: {response}")

