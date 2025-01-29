import os
from data.hm.loader import prepare_data
from data.hm.inferred_stypes import task_to_stypes
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from .model.lgbm_model import LGBMModel
from .model_pipeline import ModelPipeline
from .workflow.setup import setup_analysis_workflow

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_THREAD_LIMIT'] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'


def run_analysis(initial_data: str) -> Dict[str, List[BaseMessage]]:
    """Run the performance analysis workflow"""
    messages = [HumanMessage(content=initial_data)]
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    return workflow.run(messages)


""" 
params:
    predictions: <time, id, prediction, ground_truth...>
    model: pkl
    baseline metrics: accuracy, precision, f1, recall, log_loss...
    database: schema, attributes, features...
"""
if __name__ == "__main__":

    dataset = 'rel-hm'
    task = 'user-churn'
    subsample = 20_000
    task_params, train_df, val_df, test_df = prepare_data(
        dataset=dataset,
        task=task,
        subsample=subsample,
        init_db=False,
        generate_feats=False
    )

    full_task_name = f'{dataset}-{task}'
    col_to_stype = task_to_stypes[full_task_name]

    model = LGBMModel(task_params, col_to_stype)
    pipeline = ModelPipeline(model, col_to_stype, train_df)
    pipeline.fit(train_df)

    data = "Performance metrics show a sudden drop in model accuracy last week"
    response = run_analysis(data)
    print(f"Analysis Results: {response}")

