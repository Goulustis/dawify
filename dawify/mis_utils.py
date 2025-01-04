import rich
from rich.table import Table
import inspect

CONSOLE = rich.get_console()

def rprint(inp_str:str, print_trace:bool=True):

    frame = inspect.currentframe().f_back

    relative_file_name, line_num = "", ""
    if print_trace:
        line_num = frame.f_lineno
        file_name = frame.f_code.co_filename
        relative_file_name = file_name.split('dawify', 1)[1][1:]
    inp_str = f"[cyan]{relative_file_name}:{line_num}[/cyan]: {inp_str}"
    CONSOLE.print(inp_str)

def print_metrics(metrics: dict, header: tuple = ("Metric", "Value")):
    table = Table(title="Backtest Metrics")

    table.add_column(header[0], justify="right", style="cyan", no_wrap=True)
    table.add_column(header[1], style="magenta")

    for metric, value in metrics.items():
        table.add_row(metric, f"{value:.4f}")

    CONSOLE.print(table)