from copy import deepcopy
from matplotlib.cbook import flatten
from typing import Any, Callable
from helpers import slice_text, split_list


def _post_process_constants(scope: dict | list | tuple | set, back_track_parent=None):
    if isinstance(scope, dict):
        data = {}
        
        for k, v in scope.items():
            if v is None:
                k_new = _post_process_constants(k, back_track_parent or scope)
                
                if type(k_new) != type(k):
                    data.update(k_new)
                else:
                    data[k_new] = v
            else:
                data[k] = _post_process_constants(v, back_track_parent or scope)
        
        scope = data
    elif isinstance(scope, (list, tuple, set)):
        assert back_track_parent is not None, "Internal Error"
        
        data = scope.__class__([_post_process_constants(v, back_track_parent) for v in scope])
    elif isinstance(scope, str):
        if ";;" in scope:
            assert back_track_parent is not None, "Internal Error"
            
            parent, operation = scope.split(";;")
            
            body = _search(parent, back_track_parent)[-1]
            
            if isinstance(body, dict):
                scope = {}
                
                for k, v in body.items():
                    k_rep, v_rep = operation.split(",")
                    
                    k_rep = k_rep.format(k=k)
                    v_rep = v_rep.format(v=v)
                    
                    scope[k_rep] = v_rep
            elif isinstance(body, (list, tuple, set)):
                scope = [operation.format(v=v, i=i) for i, v in enumerate(body)]
    
    return scope

def _new_scope():
    return {
        "@strings": [],
        
        "@curly_brackets": [],
        "@circle_brackets": [],
        "@square_brackets": [],

        "@name": [],
        "@number": [],
    }

def _update_scope(scope: dict | list | tuple | set | Any, new_scope: dict | list | tuple | set | Any):
    assert type(scope) == type(new_scope), "Internal Error"
    
    if isinstance(scope, dict):
        for k, v in new_scope.items():
            scope[k] = _update_scope(scope[k], v)
    elif isinstance(scope, (list, tuple, set)):
        pass
        # for v in new_scope:
        #     if v not in scope:
        #         scope.append(v)
    else:
        if hasattr(new_scope, "__copy__"):
            new_scope = new_scope.__copy__()
        elif hasattr(new_scope, "copy"):
            new_scope = new_scope.copy()
        
        scope = new_scope
    
    return scope


def _substitute_context(
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
    
    if info_store is None:
        info_store = []
    
    placeholder_func = placeholder_func or (lambda i: f"{ {i} }")
    info_encoding_func = info_encoding_func or (lambda s: s)
    
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
        value = info_encoding_func(text[start_index + len(starter) : end_index])
        
        info_store.append(value)
        
        text = slice_text(text, start_index, end_index + len(ender), placeholder_func(info_store.index(value)))
    
    return text

def _substitute_strings(
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


def _search(value: str, scope: dict | list | tuple):
    if isinstance(scope, dict):
        for k, v in scope.items():
            if isinstance(k, str):
                if k == value:
                    return v
            
            if isinstance(v, (dict, list)):
                val_found = _search(value, v)
                if val_found is not None:
                    return k, val_found
            elif isinstance(v, tuple):
                val_found = _search(value, v[0])
                if val_found is not None:
                    return k, val_found
            elif isinstance(v, str):
                if v == value:
                    return k
    elif isinstance(scope, (list, tuple)):
        for i, v in enumerate(scope):
            if isinstance(v, (dict, list)):
                val_found = _search(value, v)
                if val_found is not None:
                    return val_found
            elif isinstance(v, tuple):
                val_found = _search(value, v[0])
                if val_found is not None:
                    return val_found
            elif isinstance(v, str):
                if v == value:
                    return i

def get_token_name(symbol: str, scope: dict, with_borders=True):
    found = _search(symbol, scope)
    
    if found is not None:
        if with_borders:
            start, end = "{", "}"
        else:
            start = end = ""
        
        return start + "?".join([str(f) for f in flatten(found)]) + end
    else:
        raise ValueError(f"'{symbol}' symbol not in scope")


LANGUAGE_CONSTANTS = _post_process_constants({
    "@keywords": [
        "var", "const", "func", "class",
        "if", "elseif", "else",
        "for", "forevery", "while",
        "get", "give", "return"
    ],
    "@loopstatments": [
        "break", "continue"
    ],
    "@specialwords": [
        "in", "at"
    ],
    "@operators": {
        "@arithmetic": {
            "@int_divide": "//",
            "@exponent": "**",
            
            "@add": "+",
            "@subtract": "-",
            "@multiply": "*",
            "@divide": "/"
        },
        "@bitwise": {
            "@bitwise_or": "|",
            "@bitwise_and": "&",
            "@bitwise_xor": "^",
            "@bitwise_modulo": "%"
        },
        "@equality": {
            "@implicit_equals": "=>",
            "@arithmetic;;{k}_by,{v}=": None,
            "@bitwise;;{k}_by,{v}=": None,
            "@equals": "="
        },
        "@comparators": {
            "@or": "||",
            "@and": "&&",
            "@xor": "^^",
            
            "@is_equal": "==",
            "@not_equal": "!=",
            
            "@greater_than": ">",
            "@less_than": "<",
            "@greater_than_or_equal": ">=",
            "@less_than_or_equal": "<=",
        },
        "@spec_bitwise": {
            "@bitwise_not": "~"
        },
        "@spec_comparators": {
            "@nor": "!",
        }
    },
    "@functionality": {
        "@dot": ".",
        "@comma": ",",
        "@colon": ":"
    },
    "@substitutions": {
        "is": "=>",
        
        "and": "&&",
        "or": "||",
        "is_equal_to": "==",
        "not_equal_to": "!=",
        
        "xor": "^^",
        "nor": "!"
    }
})


def tokenize(code: str, scope: dict[str, list] | None = None):
    scope = scope or _new_scope()
    
    code = _substitute_context(code, "//", "\n", placeholder_func=lambda _: "", strict=False)
    code = _substitute_context(code, "/*", "*/", placeholder_func=lambda _: "", strict=False)
    
    code = _substitute_strings(code, ["'", '"'], scope["@strings"], lambda i: f"`@strings?{i}`")
    code = _substitute_context(code, "{", "}", scope["@curly_brackets"], lambda i: f"{{@curly_brackets?{i}}};", lambda c: tokenize(c, scope))
    code = _substitute_context(code, "(", ")", scope["@circle_brackets"], lambda i: f"{{@circle_brackets?{i}}}", lambda c: tokenize(c, scope))
    code = _substitute_context(code, "[", "]", scope["@square_brackets"], lambda i: f"{{@square_brackets?{i}}}", lambda c: tokenize(c, scope))
    code = code.replace("`@", "{@")
    code = code.replace("`", "}")
    
    for i, s in enumerate(scope["@strings"]):
        scope["@strings"][i] = (i, s)
    
    for operator in LANGUAGE_CONSTANTS["@operators"].values():
        for symbol in filter(lambda v: len(v.strip()) == 2, operator.values()):
            code = code.replace(symbol, get_token_name(symbol, LANGUAGE_CONSTANTS))
    
    for operator in LANGUAGE_CONSTANTS["@operators"].values():
        for symbol in filter(lambda v: len(v.strip()) == 1, operator.values()):
            code = code.replace(symbol, get_token_name(symbol, LANGUAGE_CONSTANTS))
    
    replacement_data = []
    is_text = False
    is_number = False
    continue_reading = True
    
    start_i = 0
    for i, c in enumerate(code):
        if continue_reading:
            if not is_text and c.isalpha() or c == "_":
                is_text = True
                start_i = i
            elif not is_number and c.isnumeric():
                is_number = True
                start_i = i
        
        if c in (" ", ";", "{", ",", ".", ":") and continue_reading or i == len(code) - 1:
            if not is_number or c != ".":
                if c == "{":
                    continue_reading = False
                
                if is_text or is_number:
                    value_slice = slice(start_i, i + (i == len(code) - 1))
                    
                    v = code[value_slice]
                    
                    if v in LANGUAGE_CONSTANTS["@substitutions"]:
                        p_holder = get_token_name(LANGUAGE_CONSTANTS["@substitutions"][v], LANGUAGE_CONSTANTS)
                    elif next((True for k in ("@keywords", "@loopstatments", "@specialwords") if v in LANGUAGE_CONSTANTS[k]), False):
                        p_holder = get_token_name(v, LANGUAGE_CONSTANTS)
                    else:
                        if is_text:
                            if v not in scope["@name"]:
                                scope["@name"].append(v)
                        elif is_number:
                            if v not in scope["@number"]:
                                scope["@number"].append(v)
                        
                        p_holder = get_token_name(v, scope)
                        
                        if "name" not in p_holder and "number" not in p_holder:
                            print(p_holder)
                            print()
                    
                    replacement_data.insert(0, (value_slice, p_holder))
                    
                    is_text = is_number = False
            else:
                assert i != len(code) - 1, "Error"
        elif c == "}":
            continue_reading = True
    
    for value_slice, p_holder in replacement_data:
        code = slice_text(code, value_slice.start, value_slice.stop, p_holder)
    
    for symbol in LANGUAGE_CONSTANTS["@functionality"].values():
        code = code.replace(symbol, get_token_name(symbol, LANGUAGE_CONSTANTS))
    
    code_lines = [
        [v.strip().removesuffix("}")
         for v
         in line.strip().split("{")
        ][1:]
        for line
        in code.split(";") if line.strip()
    ]
    
    return code_lines, scope



