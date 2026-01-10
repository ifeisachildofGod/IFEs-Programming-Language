import inspect
from functools import wraps
from typing import Callable, Union, get_origin, get_args, Any

class OverloadException(Exception):
    pass

# Global registry: name -> {"impls": [(sig, func)], "dispatcher": callable}
_OVERLOADS = {}

def _type_matches(value, annotation) -> bool:
    # No annotation or Any: accept anything
    if annotation is inspect._empty or annotation is Any:
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional
    if origin is None:
        # Plain class/type
        try:
            return isinstance(value, annotation)
        except TypeError:
            # Annotation might be a string or typing construct not supported by isinstance
            return True  # fail-open to avoid false negatives
    elif origin is list:
        return isinstance(value, list) and all(_type_matches(v, args[0]) for v in value) if args else isinstance(value, list)
    elif origin is tuple:
        if not isinstance(value, tuple):
            return False
        if not args:
            return True
        if len(args) == 2 and args[1] is ...:  # Tuple[T, ...]
            return all(_type_matches(v, args[0]) for v in value)
        if len(args) != len(value):
            return False
        return all(_type_matches(v, t) for v, t in zip(value, args))
    elif origin is dict:
        if not isinstance(value, dict):
            return False
        if not args:
            return True
        kt, vt = args
        return all(_type_matches(k, kt) and _type_matches(v, vt) for k, v in value.items())
    elif origin is set:
        return isinstance(value, set) and all(_type_matches(v, args[0]) for v in value) if args else isinstance(value, set)
    elif origin is tuple or origin is list or origin is dict or origin is set:
        # handled above
        return True
    elif origin is type(None):
        return value is None
    elif origin is Union:
        return any(_type_matches(value, t) for t in args)
    else:
        # Other typing origins (Callable, etc.)—be permissive
        return True

def overload(func=None):
    def register(f):
        name = f.__name__
        sig = inspect.signature(f)

        entry = _OVERLOADS.get(name)
        if entry is None:
            impls = [(sig, f)]
            # Create dispatcher once per function name
            @wraps(f)
            def dispatcher(*args, **kwargs):
                candidates = _OVERLOADS[name]["impls"]

                # Try to find the first implementation that both binds and matches types
                for s, impl in candidates:
                    try:
                        bound = s.bind_partial(*args, **kwargs)
                    except TypeError:
                        continue
                    bound.apply_defaults()

                    ann = impl.__annotations__
                    ok = True
                    for param_name, value in bound.arguments.items():
                        expected = ann.get(param_name, inspect._empty)
                        if not _type_matches(value, expected):
                            ok = False
                            break

                    if ok:
                        return impl(*bound.args, **bound.kwargs)

                # If none matched strictly by type, try pure signature binding with defaults
                for s, impl in candidates:
                    try:
                        bound = s.bind_partial(*args, **kwargs)
                        bound.apply_defaults()
                        return impl(*bound.args, **bound.kwargs)
                    except TypeError:
                        continue

                raise OverloadException("No matching overload found")

            _OVERLOADS[name] = {"impls": impls, "dispatcher": dispatcher}
            return dispatcher
        else:
            # Append new implementation; return the existing dispatcher
            entry["impls"].append((sig, f))
            return entry["dispatcher"]

    # Support both @overload and @overload(...)
    if func is None:
        return register
    else:
        return register(func)

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

