import pytest
import os
from src.on_call.notebook_manager import NotebookManager


@pytest.fixture
def temp_notebook_path(tmp_path):
    """Fixture to create a temporary notebook path"""
    return str(tmp_path / "test_notebook.ipynb")


@pytest.fixture
def notebook_manager(temp_notebook_path):
    """Fixture to create a notebook manager instance"""
    manager = NotebookManager(notebook_path=temp_notebook_path)
    yield manager
    # Cleanup
    manager.shutdown()
    if os.path.exists(temp_notebook_path):
        os.remove(temp_notebook_path)


def test_notebook_creation(temp_notebook_path):
    """Test that notebook is created if it doesn't exist"""
    manager = NotebookManager(notebook_path=temp_notebook_path)
    assert os.path.exists(temp_notebook_path) == False
    manager.save_notebook()
    assert os.path.exists(temp_notebook_path)
    manager.shutdown()


def test_add_code_cell(notebook_manager):
    """Test adding a code cell to the notebook"""
    code = "print('test')"
    cell_id = notebook_manager.add_cell(code, "code")

    assert cell_id is not None
    assert len(notebook_manager.notebook.cells) == 1
    assert notebook_manager.notebook.cells[0].source == code
    assert notebook_manager.notebook.cells[0].cell_type == "code"


def test_add_markdown_cell(notebook_manager):
    """Test adding a markdown cell to the notebook"""
    markdown = "# Test Header"
    cell_id = notebook_manager.add_cell(markdown, "markdown")

    assert cell_id is not None
    assert len(notebook_manager.notebook.cells) == 1
    assert notebook_manager.notebook.cells[0].source == markdown
    assert notebook_manager.notebook.cells[0].cell_type == "markdown"


def test_update_cell(notebook_manager):
    """Test updating a cell's content"""
    # Add initial cell
    initial_code = "print('initial')"
    cell_id = notebook_manager.add_cell(initial_code, "code")

    # Update the cell
    new_code = "print('updated')"
    success = notebook_manager.update_cell(cell_id, new_code)

    assert success == True
    assert notebook_manager.notebook.cells[0].source == new_code


def test_delete_cell(notebook_manager):
    """Test deleting a cell"""
    # Add a cell
    cell_id = notebook_manager.add_cell("print('test')", "code")
    assert len(notebook_manager.notebook.cells) == 1

    # Delete the cell
    success = notebook_manager.delete_cell(cell_id)
    assert success == True
    assert len(notebook_manager.notebook.cells) == 0


def test_execute_cell(notebook_manager):
    """Test executing a single cell"""
    # Add a cell that creates a variable
    cell_id = notebook_manager.add_cell("x = 42", "code")
    notebook_manager.execute_cell(cell_id)

    # Add another cell that uses the variable
    cell_id2 = notebook_manager.add_cell("print(x)", "code")
    notebook_manager.execute_cell(cell_id2)

    # Check the output
    cell = notebook_manager.notebook.cells[1]
    assert len(cell.outputs) > 0
    assert any("42" in str(output) for output in cell.outputs)


def test_execute_all_cells(notebook_manager):
    """Test executing all cells"""
    # Add multiple cells
    notebook_manager.add_cell("a = 1", "code")
    notebook_manager.add_cell("b = 2", "code")
    notebook_manager.add_cell("print(a + b)", "code")

    # Execute all cells
    notebook_manager.execute_all_cells()

    # Check the output of the last cell
    last_cell = notebook_manager.notebook.cells[-1]
    assert len(last_cell.outputs) > 0
    assert any("3" in str(output) for output in last_cell.outputs)


def test_invalid_cell_type(notebook_manager):
    """Test adding a cell with invalid type raises ValueError"""
    with pytest.raises(ValueError):
        notebook_manager.add_cell("test", "invalid_type")


def test_kernel_shutdown(notebook_manager):
    """Test kernel shutdown"""
    assert notebook_manager.active_kernel
    notebook_manager.shutdown()
    assert not notebook_manager.active_kernel
