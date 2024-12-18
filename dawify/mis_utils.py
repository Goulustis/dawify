import rich
import inspect

CONSOLE = rich.get_console()

def rprint(inp_str:str):
    line_num = inspect.currentframe().f_back.f_lineno
    inp_str = f"[cyan]{line_num}[/cyan]: {inp_str}"
    CONSOLE.print(inp_str)