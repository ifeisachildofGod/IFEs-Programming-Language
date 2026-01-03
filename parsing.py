from matplotlib.cbook import flatten
from typing import Any, Callable
from helpers import slice_text

def _split_list(ls: list, sep: Any):
    final_list = []
    
    for v in ls:
        if v == sep:
            final_list.append([])
        else:
            final_list[-1].append(v)
    
    return final_list

def substitute_context(
        text: str,
        starter: str,
        ender: str,
        info_store: list[str | Any] | None = None,
        placeholder_func: Callable[[int], str] | None = None,
        info_encoding_func: Callable[[str], Any] | None = None,
        
        strict: bool = True
    ):
    if strict:
        assert text.count(starter) == text.count(ender), "Error"
    
    enclosements_indexes = []
    
    if placeholder_func is None:
        placeholder_func = lambda i: f"{ {i} }"
    
    if info_encoding_func is None:
        info_encoding_func = lambda s: s
    
    depth = 0
    
    prev_i = None
    for i in range(len(text)):
        c_start = text[i : i + len(starter)]
        c_end = text[i : i + len(ender)]
        
        if strict:
            if starter != ender:
                depth += (c_start == starter) - (c_end == ender)
            else:
                depth = not depth if c_start == starter else depth
            
            start_contition = c_start == starter and prev_i is None and depth == 1
            end_contition = c_end == ender and prev_i is not None and depth == 0
        else:
            start_contition = c_start == starter and prev_i is None
            end_contition = c_end == ender and prev_i is not None
        
        if start_contition:
            prev_i = i
        elif end_contition:
            enclosements_indexes.append((prev_i, i))
            prev_i = None
        
    for i, (start_index, end_index) in enumerate(reversed(enclosements_indexes)):
        if info_store is not None:
            info_store.append(info_encoding_func(text[start_index + len(starter) : end_index]))
        
        text = slice_text(text, start_index, end_index + len(ender), placeholder_func(len(enclosements_indexes) - i - 1))
    
    return text

def substitute_strings(
        text: str,
        string_chars: list[str],
        info_store: list[str | Any],
        placeholder_func: Callable[[int], str] | None = None,
        info_encoding_func: Callable[[str], Any] | None = None,
    ):
    for s in string_chars:
        assert (text.count(s) - text.count(f"\\{s}")) % 2 == 0, "Error"
    
    enclosements_indexes = []
    
    if placeholder_func is None:
        placeholder_func = lambda i: f"{ {i} }"
    
    if info_encoding_func is None:
        info_encoding_func = lambda s: s
    
    prev_i = None
    
    for i in range(len(text)):
        for s in string_chars:
            c = text[i : i + len(s)]
            string = s
            if c == s:
                break
        else:
            continue
        
        start_contition = c == string and prev_i is None
        end_contition = c == string and prev_i is not None and text[i - 1] != f"\\"
        
        if start_contition:
            prev_i = i
        elif end_contition:
            enclosements_indexes.append((prev_i, i, len(c)))
            prev_i = None
        
    for i, (start_index, end_index, l) in enumerate(reversed(enclosements_indexes)):
        sub_text_info = info_encoding_func(text[start_index + l : end_index])
        
        for s in string_chars:
            sub_text_info.replace(f"\\{s}", s)
        
        info_store.insert(0, sub_text_info)
        
        text = slice_text(text, start_index, end_index + l, placeholder_func(len(enclosements_indexes) - i - 1))
    
    return text

def single_jump_substitution(code_lines: list[list[str]], seperator: str, info_store_name: str, scope: dict[str, list]):
    for line_token in code_lines:
        if seperator in line_token:
            indexes = []
            
            sep_tracker = 1
            start_i = None
            for index, token in enumerate(line_token):
                if start_i is None and token == seperator:
                    assert index, "Error"
                    
                    start_i = index - 1
                
                if start_i is not None:
                    sep_tracker -= ((token == seperator) * 2) - 1
                    
                    if sep_tracker not in (0, 1) or sep_tracker == 1 and index == len(line_token) - 1:
                        assert sep_tracker >= 0, "Error"
                        
                        indexes.append((start_i, index + (sep_tracker == 1 and index == len(line_token) - 1)))
                        start_i = None
                        sep_tracker = 1
                
                    assert sep_tracker != 0 or index != len(line_token) - 1, "Error"
            
            for s_i, e_i in reversed(indexes):
                v = line_token[s_i : e_i].copy()
                
                for _ in range(v.count(seperator)):
                    v.remove(seperator)
                
                if v not in scope[info_store_name]:
                    index = len(scope[info_store_name])
                    scope[info_store_name].append(v)
                else:
                    index = scope[info_store_name].index(v)
                
                line_token[s_i : e_i] = [f"{info_store_name}?{index}"]

def full_cover_substitution(code_lines: list[list[str]], seperator: str, info_store_name: str, scope: dict[str, list], exception_tokens: list[str]):
    for line_token in code_lines:
        if seperator in line_token:
            s_content = line_token[:line_token.index(seperator)]
            e_content = line_token[len(line_token) - list(reversed(line_token)).index(seperator):]
            
            start_index = max(len(s_content) - list(reversed(s_content)).index(e) - 1 for e in exception_tokens if e in s_content) + 1
            end_index = len(line_token) - list(reversed(line_token)).index(seperator) + min(e_content.index(e) for e in exception_tokens if e in s_content)
            
            line_token[start_index : end_index] = [f"{info_store_name}?{len(scope[info_store_name])}"]
            scope[info_store_name].append(_split_list(line_token[start_index : end_index], seperator))
            
def search(value: str, scope: dict | list | tuple | set):
    if isinstance(scope, dict):
        for k, v in scope.items():
            if isinstance(k, str):
                if k == value:
                    return v
            
            if isinstance(v, (dict, list, tuple)):
                val_found = search(value, v)
                if val_found is not None:
                    return k, val_found
            elif isinstance(v, str):
                if v == value:
                    return k
    elif isinstance(scope, (list, tuple, set)):
        for i, v in enumerate(scope):
            if isinstance(v, (dict, list, tuple, set)):
                val_found = search(value, v)
                if val_found is not None:
                    return val_found
            elif isinstance(v, str):
                if v == value:
                    return i

def find(value: str, scope: dict):
    for key in value.split("?"):
        scope = scope[key]
    
    return scope
    

def get_token_name(symbol: str, scope: dict, with_borders=True):
    found = search(symbol, scope)
    
    if found is not None:
        if with_borders:
            start = "{"
            end = "}"
        else:
            start = end = ""
        
        return start + "?".join([str(f) for f in flatten(found)]) + end
    else:
        raise ValueError(f"'{symbol}' symbol not in scope")
