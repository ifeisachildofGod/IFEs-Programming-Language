import inspect
from functools import wraps
from typing import Callable, Union, get_origin, get_args, Any


def slice_text(text: str, start: int, end: int, value: str):
    text = list(text)
    text[start : end] = value
    
    return "".join(text)

def split_list(ls: list, sep: Any | tuple, process_func: Callable[[Any, Any | tuple], bool] | None = None):
    final_list = [[]]
    
    for v in ls:
        if process_func(v, sep) if process_func is not None else (v in sep if isinstance(sep, tuple) else v == sep):
            final_list.append([])
        else:
            final_list[-1].append(v)
    
    return final_list

