import duckdb
from jinja2 import Template
from relbench.datasets import get_dataset
from relbench.tasks import get_task

DATASET_INFO = {
    'rel-hm': {
        'tables': ['article', 'customer', 'transactions'],
        'tasks': ['user-churn', 'item-sales'],
    }
}


def db_setup(dataset_name: str, db_filename: str):
    """ Sets up a DuckDB database (at db_filename) with the tables from the specified dataset.

    Args:
        dataset_name (str): The name of the relbench dataset.
        db_filename (str): Path to the DuckDB database file.
    """
    conn = duckdb.connect(db_filename)
    dataset = get_dataset(name=dataset_name, download=True)  # noqa
    tasks = DATASET_INFO[dataset_name]['tasks']
    tables = DATASET_INFO[dataset_name]['tables']
    for table_name in tables:
        exec(f'{table_name} = dataset.get_db().table_dict["{table_name}"].df')
        conn.sql(f'create table {table_name} as select * from {table_name}')
    for task_name in tasks:
        task = get_task(dataset_name, task_name, download=True)
        train_table = task.get_table("train").df  # noqa
        val_table = task.get_table("val").df  # noqa
        test_table = task.get_table("test").df  # noqa
        task_name = task_name.replace('-', '_')
        conn.sql(f'create table {task_name}_train as select * from train_table')
        conn.sql(f'create table {task_name}_val as select * from val_table')
        conn.sql(f'create table {task_name}_test as select * from test_table')
    conn.close()


def render_jinja_sql(query: str, context: dict) -> str:
    return Template(query).render(context)
