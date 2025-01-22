import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert import HTMLExporter

from jupyter_client.manager import KernelManager
from typing import Optional, List, Dict, Any


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
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
            self.active_kernel = False

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

    def _execute_cell_object(self, cell: Dict[str, Any]):
        """
        Sends a single code cell to the kernel, captures outputs, and stores them in the notebook.
        """
        # Clear previous outputs
        cell["outputs"] = []
        cell["execution_count"] = self.execution_count

        # Create and send execute_request
        code = cell.source
        msg_id = self.kc.execute(code)

        # Collect the output messages
        while True:
            msg = self.kc.get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                # Not our message; skip.
                continue

            msg_type = msg["header"]["msg_type"]

            if msg_type == "status":
                # "status: idle" indicates the kernel finished processing
                if msg["content"]["execution_state"] == "idle":
                    break

            self._render_html()

            if msg_type in ("execute_result", "display_data"):
                # Typical outputs that go into 'outputs'
                content = msg["content"]
                output = nbformat.v4.new_output(
                    output_type=msg_type,
                    data=content["data"],
                    metadata=content["metadata"],
                    execution_count=self.execution_count,
                )
                cell["outputs"].append(output)

            elif msg_type == "stream":
                # stdout/err
                content = msg["content"]
                output = nbformat.v4.new_output(
                    output_type="stream",
                    name=content["name"],  # e.g. 'stdout' or 'stderr'
                    text=content["text"],
                )
                cell["outputs"].append(output)

            elif msg_type == "error":
                # Traceback error
                content = msg["content"]
                output = nbformat.v4.new_output(
                    output_type="error",
                    ename=content["ename"],
                    evalue=content["evalue"],
                    traceback=content["traceback"],
                )
                cell["outputs"].append(output)

        # Increment execution count globally
        self.execution_count += 1

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
        """
        Destructor: ensure kernel is closed and notebook is saved.
        """
        self.shutdown()
