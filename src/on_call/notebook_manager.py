import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert import HTMLExporter

from jupyter_client.manager import KernelManager
from typing import Optional, Dict, Any


class NotebookManager:
    """
    A notebook manager that preserves kernel state across multiple executions
    by manually managing a single kernel lifecycle through jupyter_client.
    """

    def __init__(
        self, notebook_path: str = "oncall.ipynb", kernel_name: str = "python3"
    ):
        """
        :param notebook_path: Where to load/save the notebook.
        :param kernel_name:   Which kernel to launch (default "python3").
        """
        self.notebook_path = notebook_path
        self.kernel_name = kernel_name

        # Load existing notebook or create a new one
        if os.path.exists(self.notebook_path):
            self.notebook = nbformat.read(self.notebook_path, as_version=4)
        else:
            self.notebook = new_notebook()

        # Keep track of global execution_count for the notebook
        # We'll manually increment it each time we execute a cell.
        # Alternatively, you can read the highest cell execution_count from the notebook itself.
        self.execution_count = 1

        # Start the kernel
        self.km, self.kc = self._start_kernel()

        # Flag to track if we have an active kernel
        self.active_kernel = True

    def _start_kernel(self):
        """
        Starts a kernel manager and client for the specified kernel_name.
        Returns (kernel_manager, kernel_client).
        """
        km = KernelManager(kernel_name=self.kernel_name)
        km.start_kernel()
        kc = km.client()
        kc.start_channels()

        # Optional: Wait until the kernel is fully ready
        kc.wait_for_ready()

        return km, kc

    def _stop_kernel(self):
        """
        Gracefully stops the kernel and its channels.
        """
        if self.active_kernel:
            try:
                self.kc.stop_channels()
                self.km.shutdown_kernel(now=True)
                del self.kc
                del self.km
                self.active_kernel = False
            except:
                pass

    def add_cell(self, source: str, cell_type: str = "code") -> str:
        """
        Add a new cell (code or markdown) to the notebook in memory.
        :return: The assigned cell ID.
        """
        if cell_type not in ("code", "markdown"):
            raise ValueError("cell_type must be 'code' or 'markdown'.")

        if cell_type == "code":
            new_cell = new_code_cell(source=source)
        else:
            new_cell = new_markdown_cell(source=source)

        self.notebook.cells.append(new_cell)
        return new_cell["id"]

    def _find_cell_index_by_id(self, cell_id: str) -> int:
        """
        Finds the index of a cell by its ID in the notebook.
        Returns -1 if not found.
        """
        for idx, cell in enumerate(self.notebook.cells):
            if cell.get("id", None) == cell_id:
                return idx
        return -1

    def update_cell(self, cell_id: str, new_source: str) -> bool:
        """
        Updates the source of an existing cell.
        """
        idx = self._find_cell_index_by_id(cell_id)
        if idx == -1:
            return False
        self.notebook.cells[idx].source = new_source
        return True

    def delete_cell(self, cell_id: str) -> bool:
        """
        Deletes a cell from the notebook.
        """
        idx = self._find_cell_index_by_id(cell_id)
        if idx == -1:
            return False
        self.notebook.cells.pop(idx)
        return True

    def execute_all_cells(self):
        """
        Execute every code cell in the notebook in order, preserving kernel state.
        """
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self._execute_cell_object(cell)

    def execute_cell(self, cell_id: Optional[str] = None):
        """
        Execute a single code cell by ID. If no ID is given, execute the last code cell.
        """
        # Find target cell
        if cell_id is None:
            code_cells = [c for c in self.notebook.cells if c.cell_type == "code"]
            if not code_cells:
                print("No code cells to execute.")
                return
            cell = code_cells[-1]  # last code cell
        else:
            idx = self._find_cell_index_by_id(cell_id)
            if idx == -1:
                print(f"Cell with id={cell_id} not found.")
                return
            cell = self.notebook.cells[idx]
            if cell.cell_type != "code":
                print(f"Cell {cell_id} is not a code cell.")
                return

        self._execute_cell_object(cell)

    def get_cell_outputs(self, cell_id: str):
        idx = self._find_cell_index_by_id(cell_id)
        if idx == -1:
            print(f"Cell with id={cell_id} not found.")
            return None

        cell = self.notebook.cells[idx]
        if "outputs" not in cell:
            return None

        for output in cell["outputs"]:
            self._display_output(output)

        return cell["outputs"]

    def _display_output(self, output):
        output_handlers = {
            'stream': lambda o: print(f"{o['name']}: {o['text']}", end=''),
            'execute_result': lambda o: print(o['data'].get('text/plain', '')),
            'error': lambda o: print('\n'.join(o['traceback'])),
            'display_data': lambda o: print(o['data'].get('text/plain', ''))
        }

        handler = output_handlers.get(output['output_type'])
        if handler:
            handler(output)

    def _create_output(self, msg_type: str, content: dict, execution_count: int = None) -> dict:
        if msg_type in ("execute_result", "display_data"):
            return nbformat.v4.new_output(
                output_type=msg_type,
                data=content["data"],
                metadata=content["metadata"],
                execution_count=execution_count
            )
        elif msg_type == "stream":
            return nbformat.v4.new_output(
                output_type="stream",
                name=content["name"],
                text=content["text"]
            )
        elif msg_type == "error":
            return nbformat.v4.new_output(
                output_type="error",
                ename=content["ename"],
                evalue=content["evalue"],
                traceback=content["traceback"]
            )
        return {}

    def _execute_cell_object(self, cell: Dict[str, Any]):
        cell["outputs"] = []
        cell["execution_count"] = self.execution_count

        code = cell["source"]
        msg_id = self.kc.execute(code)

        print(f"\nIn [{self.execution_count}]:")
        print(code)
        print(f"\nOut[{self.execution_count}]:")

        while True:
            msg = self.kc.get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            msg_type = msg["header"]["msg_type"]
            if msg_type == "status" and msg["content"]["execution_state"] == "idle":
                break
            if msg_type in ("execute_result", "display_data", "stream", "error"):
                output = self._create_output(
                    msg_type,
                    msg["content"],
                    self.execution_count
                )
                if output:
                    cell["outputs"].append(output)
                    self._display_output(output)

        self._render_html()
        self.save_notebook()
        self.execution_count += 1
        print("\n")

    def save_notebook(self):
        """
        Writes the notebook to the file system.
        """
        nbformat.write(self.notebook, self.notebook_path)

    def shutdown(self):
        """
        Stops the kernel and saves the notebook to disk.
        Call this when you're done with the notebook manager.
        """
        self._stop_kernel()
        self.save_notebook()

    def _render_html(self):
        (body, _) = HTMLExporter().from_notebook_node(self.notebook)
        with open(f"{self.notebook_path.split('.')[0]}.html", "w", encoding="utf-8") as f:
            f.write(body)

    def __del__(self):
        pass
