
from typing import Any, Callable, Literal
from parse_token import ParsedCode


class PythonLinker:
    @staticmethod
    def call(_, __, c_class: "ParsedCode", parameters: "ParsedCode", **kwargs):
        
        if c_class.variables["name"] == "_raw":
            return parameters.variables["value"]
        
        params = []
        if parameters is not None:
            params = parameters.variables["collection"] if "collection" in parameters.variables else [parameters]
        
        return f"self.{c_class.variables["name"]}({", ".join([varias.variables["name"] if "name" in varias.variables else (f'"{varias.variables["value"]}"' if isinstance(varias.variables["value"], str) else varias.variables["value"]) for varias in params])})"
    
    @staticmethod
    def function(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            name: str,
            params: "ParsedCode",
            action: list["ParsedCode"],
            **kwargs
        ):
        
        depth: int = kwargs.get("depth", 1)
        
        action_texts = [str(_link(c, depth=depth + 1)) for c in action] if action else ["pass"]
        
        return f"""def {name}(self, {", ".join(
            [
                str(_link(varias, ))
                for varias in
                (params.variables["collection"] if params.type == "COLLECTION" else [params])
            ]
            )}):\n{"\n".join([f"{"\t" * depth}{t}" for t in action_texts])}\n"""
    
    @staticmethod
    def cls(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            name: str,
            donor: "ParsedCode | None",
            action: list["ParsedCode"],
            scope: dict[str, dict[str, "ParsedCode"]],
            **kwargs
        ):
        
        donor_text = ""
        
        if donor is not None:
            _donor_list = [donors.variables["name"] for donors in (donor.variables["collection"] if donor.type == "COLLECTION" else [donor])]
            donor_text = f"({", ".join(_donor_list)})"
        
        if name in scope["func"]:
            init_func = scope["func"]["__init__"] = scope["func"].pop(name)
            init_func.variables["name"] = "__init__"
        
        depth: int = kwargs.get("depth", 1)
        action_texts = [str(_link(c, depth=depth + 1)) for c in action] if action else ["pass"]
        
        return f"class {name}{donor_text}:\
            \n{"\n".join([f"{"\t" * depth}{t}" for t in action_texts])}\n"

    @staticmethod
    def definition(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            name: str,
            **kwargs
        ):
        return f'{name}: "{data_type.variables["name"]}"'
    
    @staticmethod
    def assignment(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            object: "ParsedCode",
            value: "ParsedCode | None",
            **kwargs
        ):
        return f"{_link(object)}{f' = {_link(value)}' if value else ''}"
    
    @staticmethod
    def c_definition(*args, **kwargs):
        return PythonLinker.definition(*args, **kwargs)
    
    @staticmethod
    def c_assignment(*args, **kwargs):
        return PythonLinker.assignment(*args, **kwargs)
    
    @staticmethod
    def string_lit(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            value: str,
            **kwargs
        ):
        return f'"{value}"'
    
    @staticmethod
    def number_lit(
            context: dict[str, dict[str, "ParsedCode"]],
            data_type: "ParsedCode | None",
            value: str,
            **kwargs
        ):
        return f'{value}'

class CLinker:
    pass


P_FUNTIONS = {}

def link(code: list[ParsedCode], fmt: Literal["py", "c"], l_proc: Callable[[list[Any]], Any] | None = None):
    global P_FUNTIONS
    
    match fmt:
        case "py":
            exec_class = PythonLinker()
        case "c":
            exec_class = CLinker()
    
    
    for k in set(exec_class.__class__.__dict__):
        value = exec_class.__class__.__dict__[k]
        
        if k.startswith("_") or not isinstance(value, staticmethod):
            continue
        
        P_FUNTIONS[k] = value
    
    l_proc = l_proc or (lambda l: "\n".join(l))
    
    return l_proc([_link(p_code) for p_code in code])

def _link(p_code: ParsedCode, **kwargs):
    global P_FUNTIONS
    
    k = p_code.type.lower()
    
    if k not in P_FUNTIONS:
        return

    kwargs.update(p_code.variables)
    
    return P_FUNTIONS[k](p_code.scope_data, p_code.data_type, **kwargs)

