from dotenv import load_dotenv
import os
import sys

if not __package__:
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)

from on_call.data.hm.loader import fetch_data, TimeConfig
from on_call.data.hm.inferred_stypes import task_to_stypes
from on_call.model.lgbm_model import LGBMModel
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


def run_analysis(_pipeline: ModelPipeline) -> LangGraphMessageState:
    state = {"pipeline": _pipeline}
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    result = workflow.run(state)
    return result


if __name__ == "__main__":
    custom_time = TimeConfig(year=2020, train_months=range(1, 7), val_month=7, test_month=8)
    task_params, train_df, val_df, test_df = fetch_data(
        dataset=DATASET, task=TASK, subsample=SUBSAMPLE, time_config=custom_time)

    pipeline = ModelPipeline.load(PIPELINE_PATH)
    pipeline.enrich_with_metrics(val_df)

    resultant_test_df = pipeline.predict_and_append_to_df(test_df)
    pipeline.enrich_with_ref_data(train_df=train_df, val_df=val_df, test_df=resultant_test_df)

    response = run_analysis(pipeline)
    print(f"Analysis Results: {response}")

