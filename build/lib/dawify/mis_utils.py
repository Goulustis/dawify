import rich
import inspect

CONSOLE = rich.get_console()

def rprint(inp_str:str):

    frame = inspect.currentframe().f_back
    file_name = frame.f_code.co_filename
    relative_file_name = file_name.split('dawify', 1)[1][1:]
    line_num = frame.f_lineno
    inp_str = f"[cyan]{relative_file_name}:{line_num}[/cyan]: {inp_str}"
    CONSOLE.print(inp_str)