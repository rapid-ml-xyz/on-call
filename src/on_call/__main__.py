from dotenv import load_dotenv
import argparse
import joblib
import pandas as pd
import os
import sys

if not __package__:
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)

from on_call.notebook_manager import NotebookManager
from on_call.model.random_forest_regressor_model import RandomForestRegressor
from on_call.orchestrator.engines import LangGraphMessageState
from on_call.workflow.setup import setup_analysis_workflow

DEFAULT_MODEL_PATH = 'joblibs/random_forest_bike_sharing.joblib'
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='Run on-call analysis with specified dataset')
    parser.add_argument('--data-path', type=str, default='data/bike_sharing_data.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--model-path', type=str,
                        help='Path to saved model. If not provided, a new model will be built')
    return parser.parse_args()


def build_model(raw_data, save_path=DEFAULT_MODEL_PATH):
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    return model


def run_analysis(model, notebook_manager) -> LangGraphMessageState:
    state = {
        "model": model,
        "notebook_manager": notebook_manager
    }
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    result = workflow.run(state)
    return result


if __name__ == "__main__":
    args = parse_args()

    if args.model_path:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model not found at path: {args.model_path}")
        _model = joblib.load(args.model_path)
    else:
        df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        _model = build_model(df)

    _notebook_manager = NotebookManager(notebook_path="oncall.ipynb")
    try:
        response = run_analysis(_model, _notebook_manager)
    finally:
        _notebook_manager.shutdown()
