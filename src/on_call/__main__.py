import os
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from .model.gbdt_model import get_model
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


if __name__ == "__main__":
    """ 
    params:
        predictions: <time, id, prediction, ground_truth...>
        model: pkl
        baseline metrics: accuracy, precision, f1, recall, log_loss...
        database: schema, attributes, features...
    """

    # TODO: Come up with a dynamic way of passing True for init_db & generate_feats first time round
    gbdt, val_metrics, test_pred = get_model(
        dataset='rel-hm', task='user-churn', subsample=20_000, init_db=False, generate_feats=False)
    print(val_metrics)

    data = "Performance metrics show a sudden drop in model accuracy last week"
    response = run_analysis(data)
    print(f"Analysis Results: {response}")

