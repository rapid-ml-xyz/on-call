import os
from jinja2 import Template
from on_call.modules.base_module import BaseModule
from on_call.orchestrator import WorkflowState


class ImportsModule(BaseModule):
    def run(self) -> WorkflowState:
        notebook_manager = self.state['notebook_manager']

        template_path = os.path.join(os.path.dirname(__file__), 'template.jinja')
        with open(template_path) as f:
            imports_template = Template(f.read())

        template_vars = {
            'custom_imports': '',
            'startup_message': 'All modules imported!'
        }

        setup_code = imports_template.render(**template_vars)
        setup_cell_id = notebook_manager.add_cell(setup_code)
        notebook_manager.execute_cell(setup_cell_id)

        return WorkflowState(self.state)
