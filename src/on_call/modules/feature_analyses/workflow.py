import os
from jinja2 import Template
from on_call.model.random_forest_regressor_model import RandomForestRegressor
from on_call.modules.base_module import BaseModule
from on_call.orchestrator import WorkflowState


class FeatureAnalyses(BaseModule):
    def run(self) -> WorkflowState:
        model: RandomForestRegressor = self.state['model']
        notebook_manager = self.state['notebook_manager']

        template_path = os.path.join(os.path.dirname(__file__), 'template.jinja')
        with open(template_path) as f:
            template = Template(f.read())

        template_vars = {
            'numerical_features': str(model.numerical_features),
            'categorical_features': str(model.categorical_features)
        }

        cells = template.render(**template_vars).split('---CELL---')
        for cell in cells:
            if cell.strip():
                cell_id = notebook_manager.add_cell(cell.strip())
                notebook_manager.execute_cell(cell_id)

        return WorkflowState(self.state)
