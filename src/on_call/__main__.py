from dotenv import load_dotenv
from data.hm.loader import fetch_data
from data.hm.inferred_stypes import task_to_stypes
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from .model.lgbm_model import LGBMModel
from .model_pipeline import ModelPipeline
from .workflow.setup import setup_analysis_workflow

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


def run_analysis(initial_data: str) -> Dict[str, List[BaseMessage]]:
    """Run the performance analysis workflow"""
    messages = {
        "messages": initial_data,
        "foo": "bar",
        "baz": "qux"
    }
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    return workflow.run(messages)


if __name__ == "__main__":
    task_params, train_df, val_df, test_df = fetch_data(DATASET, TASK, SUBSAMPLE)

    pipeline = ModelPipeline.load(PIPELINE_PATH)
    pipeline.enrich_with_metrics(val_df)

    resultant_test_df = pipeline.predict_and_append_to_df(test_df)
    pipeline.enrich_with_ref_data(train_df=train_df, val_df=val_df, test_df=resultant_test_df)

    data = "Performance metrics show a sudden drop in model accuracy last week"
    response = run_analysis(data)
    print(f"Analysis Results: {response}")

