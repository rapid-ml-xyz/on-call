import os
from jinja2 import Template
from on_call.model.random_forest_regressor_model import RandomForestRegressor
from on_call.modules.base_module import BaseModule
from on_call.orchestrator import WorkflowState


class TimeSeriesReports(BaseModule):
    def run(self) -> WorkflowState:
        model: RandomForestRegressor = self.state['model']
        notebook_manager = self.state['notebook_manager']

        model.current_data.to_csv('current.csv')
        model.reference_data.to_csv('reference.csv')

        load_current_df = notebook_manager.add_cell(
            "current = pd.read_csv('current.csv', index_col=0, parse_dates=True)")
        notebook_manager.execute_cell(load_current_df)

        load_reference_df = notebook_manager.add_cell(
            "reference = pd.read_csv('reference.csv', index_col=0, parse_dates=True)")
        notebook_manager.execute_cell(load_reference_df)

        template_path = os.path.join(os.path.dirname(__file__), 'template.jinja')
        with open(template_path) as f:
            template = Template(f.read())

        template_vars = {
            'target': f"'{model.target}'",
            'prediction': f"'{model.prediction_col}'",
            'numerical_features': str(model.numerical_features),
            'categorical_features': str(model.categorical_features)
        }

        cells = template.render(**template_vars).split('---CELL---')
        for cell in cells:
            if cell.strip():
                cell_id = notebook_manager.add_cell(cell.strip())
                notebook_manager.execute_cell(cell_id)

        return WorkflowState(self.state)
