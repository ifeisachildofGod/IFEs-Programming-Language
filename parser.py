

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from lexer import *

@dataclass
class ParsedCode:
    type: str
    data: dict[str, Any]
    data_type: "ParsedCode" = None
    
    def copy(self):
        return ParsedCode(type=self.type, data=self.data, data_type=self.data_type)
    
    def update(self, other: "ParsedCode"):
        self.type = other.type
        self.data = other.data
        self.data_type = other.data_type


PARSED_CODE: list[ParsedCode] = []
SCOPE: dict[str, list] | None = None


def _search(value: str, scope: dict | list | tuple | set):
    if isinstance(scope, dict):
        for k, v in scope.items():
            if isinstance(k, str):
                if k == value:
                    return v
            
            if isinstance(v, (dict, list, tuple)):
                val_found = _search(value, v)
                if val_found is not None:
                    return k, val_found
            elif isinstance(v, str):
                if v == value:
                    return k
    elif isinstance(scope, (list, tuple, set)):
        for i, v in enumerate(scope):
            if isinstance(v, (dict, list, tuple, set)):
                val_found = _search(value, v)
                if val_found is not None:
                    return val_found
            elif isinstance(v, str):
                if v == value:
                    return i


def token_type(token: str):
    return token.split("?")[0]

def token_value(token: str):
    scope = LANGUAGE_CONSTANTS
    if SCOPE is not None:
        scope.update(SCOPE)
    
    for key in token.split("?"):
        if isinstance(scope, list):
            key = int(key)
        elif isinstance(scope, tuple):
            scope = scope[0]
        
        scope: dict | list | tuple | str = scope[key]
    
    return scope

def get_token_name(symbol: str, with_borders=False):
    scope = LANGUAGE_CONSTANTS
    scope.update(SCOPE if SCOPE is not None else {})
    
    found = _search(symbol, scope)
    
    if found is not None:
        if with_borders:
            start = "{"
            end = "}"
        else:
            start = end = ""
        
        return start + "?".join([str(f) for f in flatten(found)]) + end
    else:
        raise ValueError(f"'{symbol}' symbol not in scope")


# def is_defined(tokens: list[str], data: dict[dict[str, ParsedCode]]):
#     variable = parse_line(tokens, data)
    
#     if isinstance(variable, ParsedCode):
#         if variable.type == "REFERENCE":
#             if next((False for token in variable.data["line"] if token.type in ("INDEX", "CALL")), True):
#                 for index, code_point in enumerate(variable.data["line"]):
#                     if code_point.type == "NAME":
#                         code_point.data_type
#                         variable.data["line"][index - 1]
            
#             return True
        
#         return variable.type in ("DEFINITION", "CLASS")
    
#     raise Exception("Value is not a variable")


def is_token_type(token: str, _type: str | tuple):
    if isinstance(_type, tuple):
        for correct in _type:
            if is_token_type(token, correct):
                break
        else:
            return False
        
        return True
    elif isinstance(_type, str):
        return token.split("?")[:_type.count("?") + 1] == _type.split("?")
    else:
        raise Exception("Invalid type passed")

def assert_token_type(token: str, _type: str | tuple, is_not=False, error_msg: str | None = None):
    assert is_token_type(token, _type) ^ is_not, error_msg


def eat(name: str, code_lines: list[ParsedCode] | None = None, data_type=None, **variables):
    code_line = ParsedCode(name, variables, data_type)
    
    if code_lines is not None:
        code_lines.append(code_line)
    
    return code_line


def parse_line(tokens: list[str], line: int, data: dict[str, dict[str, ParsedCode]], code_lines: list[ParsedCode] | None = None, scope: dict[str, list] | None = None, parents: list | None=None, a0=None):
    global SCOPE
    
    prev_scope = SCOPE
    
    if scope is not None:
        SCOPE = scope
    if code_lines is None:
        code_lines = []
    
    comma_present = get_token_name(",") in tokens
    keyword_present = is_token_type(tokens[0], "@keywords")
    equality_present = get_token_name("=") in tokens
    equality_by_index = next((i for i, t_v in enumerate(tokens) if is_token_type(t_v, "@operators?@equality")), None)
    arith_or_bit_or_comp_value = next(((i, t) for i, t in enumerate(tokens) if is_token_type(t, ("@operators?@arithmetic", "@operators?@bitwise", "@operators?@comparators"))), None)
    spec_bit_or_comp_present = next((True for t in tokens if is_token_type(t, ("@operators?@spec_bitwise", "@operators?@spec_comparators"))), False)
    dot_present = get_token_name(".") in tokens
    circle_brackets_end_index = next((len(tokens) - i - 1 for i, t_v in enumerate(reversed(tokens)) if is_token_type(t_v, "@circle_brackets")), None)
    square_brackets_end_index = next((len(tokens) - i - 1 for i, t_v in enumerate(reversed(tokens)) if is_token_type(t_v, "@square_brackets")), None)
    token_name_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@name") else None
    token_ls_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@loopstatments") else None
    token_number_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@number") else None
    token_string_value = token_value(tokens[0])[1] if len(tokens) == 1 and is_token_type(tokens[0], "@strings") else None
    
    code_value = None
    
    if comma_present:
        code_value = eat("COLLECTION", code_lines, collection=[parse_line(token, line, data) for token in split_list(tokens, get_token_name(","))])
    elif keyword_present:
        match token_value(tokens[0]):
            case "var":
                if get_token_name("=") in tokens:
                    definition, value = split_list(tokens, get_token_name("="))
                    (_, name), _type = split_list(definition, get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg=f"Error on line {line}")
                    # assert not is_defined(name, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
                    
                    code_value = data["variables"][token_value(name)] = eat("DEFINITION", code_lines, data_type=parse_line(_type, line, data), parent=parents)
                    
                    eat("ASSIGNMENT", code_lines, object=code_value, value=parse_line(value, line, data))
                else:
                    (_, name), _type = split_list(tokens, get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    # assert not is_defined(name, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
                    
                    code_value = data["variables"][token_value(name)] = eat("DEFINITION", type=parse_line(_type, line, data), parent=parents)
            case "const":
                if a0 == "FUNC":
                    (_, name), _type = split_list(tokens, get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    # assert not is_defined(name, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
                    
                    code_value = data["constants"][token_value(name)] = eat("C-DEFINITION", code_lines, data_type=parse_line(_type, line, data), parent=parents)
                else:
                    definition, value = split_list(tokens, get_token_name("="))
                    (_, name), _type = split_list(definition, get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    # assert not is_defined(name, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
                    
                    code_value = data["constants"][token_value(name)] = eat("C-DEFINITION", code_lines, data_type=parse_line(_type, line, data), parent=parents)
                    
                    eat("C-ASSIGNMENT", code_lines, object=code_value, value=parse_line(value, line, data))
            case "func":
                if len(tokens) == 4:
                    _, name, parameters, action = tokens
                    
                    name = token_value(name)
                    
                    assert name == "main", f"Error on line {line}"
                    
                    assert_token_type(parameters, "@circle_brackets", error_msg=f"Error on line {line}")
                    parameters, param_scope = token_value(parameters)
                    assert len(parameters) == 1, f"Error on line {line}"
                    parameters = parameters[0]
                    
                    assert_token_type(action, "@curly_brackets", error_msg=f"Error on line {line}")
                    
                    return_type = data["classes"]["int"]
                    
                    func_actions = []
                    sub_data = deepcopy(data)
                    action, action_scope = token_value(action)
                    for l, token in enumerate(action):
                        parse_line(token, line + l + 1, sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)])
                    
                    data["functions"][name] = code_value = eat("MAIN", code_lines, params=parse_line(parameters, line, sub_data, scope=param_scope, parents=parents + [(code_value, return_type)], a0="FUNC"), action=func_actions)
                else:
                    if len(split_list(tokens, get_token_name("=>"))[0]) == 3:
                        (_, name, parameters), part2 = split_list(tokens, get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, data), part2[-1]
                    elif len(split_list(tokens, get_token_name("=>"))[0]) == 4:
                        (_, name, generics, parameters), part2 = split_list(tokens, get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, data), part2[-1]
                        
                        assert_token_type(generics, "@square_brackets", error_msg = f"Error on line {line}")
                        generics, generics_scope = token_value(generics)
                        assert len(generics) == 1, f"Error on line {line}"
                        generics = parse_line(generics[0], line, deepcopy(data), scope=generics_scope, a0=True)
                        generic_collection = generics.collection if generics.type == "COLLECITON" else [generics]
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    
                    assert_token_type(parameters, "@circle_brackets", error_msg = f"Error on line {line}")
                    parameters, param_scope = token_value(parameters)
                    assert len(parameters) == 1, f"Error on line {line}"
                    parameters = parameters[0]
                    
                    assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                    
                    data["functions"][token_value(name)] = code_value = eat("FUNCTION", code_lines, return_type=return_type, data_type="FUNCTION")
                    
                    sub_data = deepcopy(data)
                    
                    if len(split_list(tokens, get_token_name("=>"))[0]) == 4:
                        for g in generic_collection:
                            sub_data["constants"][g.data["value"]] = eat("GENERIC", [], data_type="GENERIC")
                    
                    func_actions = []
                    action, action_scope = token_value(action)
                    for l, token in enumerate(action):
                        parse_line(token, line + l + 1, sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)])
                    
                    code_value.data["params"] = parse_line(parameters, line, sub_data, scope=param_scope, parents=parents + [(code_value, return_type)], a0="FUNC")
                    code_value.data["action"] = func_actions
                    code_value.data["scope"] = sub_data
            case "class":
                if len(tokens) == 4:
                    _, name, donor, action = tokens
                elif len(tokens) == 5:
                    _, name, generics, donor, action = tokens
                    
                    assert_token_type(generics, "@square_brackets", error_msg = f"Error on line {line}")
                    generics, generics_scope = token_value(generics)
                    assert len(generics) == 1, f"Error on line {line}"
                    generics = parse_line(generics[0], line, data, scope=generics_scope, a0=True)
                    generic_collection = generics.collection if generics.type == "COLLECITON" else [generics]
                
                assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                
                assert_token_type(donor, "@circle_brackets", error_msg = f"Error on line {line}")
                donor, donor_scope = token_value(donor)
                assert len(donor) <= 1, f"Error on line {line}"
                donor = parse_line(donor[0], line, data, scope=donor_scope) if donor else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                name = token_value(name)
                
                code_value = data["classes"][name] = eat("CLASS", code_lines, donor=donor, data_type="CLASS")
                
                sub_data = deepcopy(data)
                
                if len(tokens) == 5:
                    for g in generic_collection:
                        sub_data["constants"][g.data["value"]] = eat("GENERIC", [], data_type="GENERIC")
                
                class_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parsed_line = parse_line(token, line + l + 1, sub_data, class_actions, action_scope, parents=parents + [code_value])
                    assert parsed_line.type in ("CLASS", "FUNCTION", "DEFINITION", "IMPORT"), f"Error on line {line + l + 1}"
                
                code_value.data["action"] = class_actions
                code_value.data["scope"] = sub_data
            
            case "for":
                _, conditions, action = tokens
                
                assert_token_type(conditions, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = deepcopy(data)
                
                loop_conditions = []
                conditions, condition_scope = token_value(conditions)
                for l, token in enumerate(conditions):
                    parse_line(token, line + l + 1, sub_data, loop_conditions, condition_scope)
                
                code_value = eat("FOR-LOOP", code_lines, conditions=loop_conditions)
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, sub_data, loop_actions, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = loop_actions
            case "forevery":
                (_, variables), part2 = split_list(tokens, get_token_name("in"))
                collection, action = part2[:-1], part2[-1]
                
                assert_token_type(variables, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = deepcopy(data)
                
                variables, variables_scope = token_value(variables)
                assert len(variables) == 1, "Error"
                variables = parse_line(variables[0], line, sub_data, scope=variables_scope)
                
                code_value = eat("FOR-EVERY-LOOP", code_lines, variables=variables, collection=parse_line(collection, line, data))
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, sub_data, loop_actions, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = loop_actions
            case "while":
                _, condition, action = tokens
                
                assert_token_type(condition, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = deepcopy(data)
                
                condition, condition_scope = token_value(condition)
                assert len(condition) == 1, "Error"
                condition = condition[0]
                
                code_value = eat("WHILE-LOOP", code_lines, condition=parse_line(condition, line, sub_data, scope=condition_scope))
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, sub_data, loop_actions, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = loop_actions
            
            case "if":
                _, condition, action = tokens
                
                sub_data = deepcopy(data)
                
                assert_token_type(condition, "@circle_brackets", error_msg = f"Error on line {line}")
                condition, condition_scope = token_value(condition)
                assert len(condition) <= 1, f"Error on line {line}"
                condition = parse_line(condition[0], line, sub_data, scope=condition_scope) if condition else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("IF", code_lines, condition=condition)
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, sub_data, conditonal_actions, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = conditonal_actions
            case "elseif":
                assert code_lines[-1].type in ("IF", "ELSE-IF")
                
                _, condition, action = tokens
                
                sub_data = deepcopy(data)
                
                assert_token_type(condition, "cicle_brackets", error_msg = f"Error on line {line}")
                condition, condition_scope = token_value(condition)
                assert len(condition) == 1, f"Error on line {line}"
                condition = parse_line(condition[0], line, sub_data, scope=condition_scope) if condition else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("ELSE-IF", code_lines, parent=code_lines[-1], condition=condition)
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, sub_data, conditonal_actions, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = conditonal_actions
            case "else":
                assert code_lines[-1].type in ("IF", "ELSE-IF")
                
                _, action = tokens
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("ELSE", code_lines, parent=code_lines[-1])
                
                sub_data = deepcopy(data)
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, conditonal_actions, sub_data, action_scope, parents=parents + [code_value])
                
                code_value.data["action"] = conditonal_actions
            
            case "get":
                code_value = eat("IMPORT", code_lines, module=parse_line(tokens[1:], line, data, a0=True))
            case "give":
                code_value = eat("EXPORT", code_lines, cls=parse_line(tokens[1:], line, data))
            
            case "return":
                return_value = parse_line(tokens[1:], line, data)
                
                assert return_value.data_type == next(type_hierachy[1] for type_hierachy in reversed(parents) if isinstance(type_hierachy, (list, tuple))), f"Error on line {line}"
                
                code_value = eat("RETURN", code_lines, value=return_value)
        
        if code_value is None:
            raise Exception("Internal Error")
    elif equality_present:
        definition, value = split_list(tokens, get_token_name("="))
        
        # assert is_defined(definition, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
        
        code_value = eat("ASSIGNMENT", code_lines, object=parse_line(definition, line, data), value=parse_line(value, line, data))
    elif equality_by_index is not None:
        assert len([i for i, t_v in enumerate(tokens) if is_token_type(t_v, "@operators?@equality")]), f"Error on line {line}"
        
        equality_type = tokens[equality_by_index]
        
        variable, value = split_list(tokens, equality_type)
        
        value = parse_line(value, line, data)
        
        if "@implicit_equals" in equality_type:
            assert len(variable) == 1, f"Error on line {line}"
            
            name = variable[0]
            
            # assert not is_defined(name, data["variables"], data["constants"], data["classes"]), f"Error on line {line}"
            assert is_token_type(name, "@name"), f"Error on line {line}"
            
            data["variables"][token_value(name)] = eat("DEFINITION", code_lines, type=value.data_type)
            
            code_value = eat("ASSIGNMENT", code_lines, object=data["variables"][name], value=value)
        else:
            # assert is_defined(variable, data["variables"], data["classes"]), f"Error on line {line}"
            
            code_value = eat("ASSIGNMENT-BY", code_lines, object=parse_line(variable, line, data), value=value)
    elif arith_or_bit_or_comp_value is not None:
        first_arith_bit_index, arith_bit_token = arith_or_bit_or_comp_value
        
        ls = parse_line(tokens[:first_arith_bit_index], line, data)
        rs = parse_line(tokens[first_arith_bit_index + 1:], line, data)
        
        code_value = eat("BIN-OP", code_lines, left_side=ls, right_side=rs, op_type=arith_bit_token)
    elif spec_bit_or_comp_present:
        assert is_token_type(tokens[0], ("@operators?@spec_bitwise", "@operators?@spec_comparators")), f"Error on line {line}"
        
        code_value = eat("SPEC-OP", code_lines, target=parse_line(tokens[1:], line, data), op_type=tokens[0])
    elif dot_present:
        ref = None
        line_content = split_list(tokens, get_token_name("."))
        
        for obj in line_content:
            ref = parse_line(obj, line, ref.data_type.data["scope"] if ref is not None else data, code_lines, scope, a0=True)
        
        code_value = ref
    elif circle_brackets_end_index is not None and (square_brackets_end_index is None or circle_brackets_end_index > square_brackets_end_index):
        if len(tokens) == 1:
            value, value_scope = token_value(tokens[0])
            assert len(value) == 1, f"Error on line {line}"
            
            code_value = parse_line(value[0], line, data, scope=value_scope)
        else:
            assert tokens[:circle_brackets_end_index], f"Error on line {line}"
            
            params, params_scope = token_value(tokens[circle_brackets_end_index])
            assert len(params) <= 1, f"Error on line {line}"
            parameters = parse_line(params[0], line, data, scope=params_scope) if params else None
            
            callable_value = parse_line(tokens[:circle_brackets_end_index], line, data)
            
            code_value = eat("CALL", code_lines, callable=callable_value, parameters=parameters, data_type=callable_value.data["return_type"])
    elif square_brackets_end_index is not None and (circle_brackets_end_index is None or square_brackets_end_index > circle_brackets_end_index):
        if len(tokens) == 1:
            ls, ls_scope = token_value(tokens[0])
            assert len(ls) <= 1, f"Error on line {line}"
            collection = parse_line(ls[0], line, data, scope=ls_scope) if ls else None
            
            code_value = eat("LIST", code_lines, collection=collection)
        else:
            assert tokens[:square_brackets_end_index], f"Error on line {line}"
            
            index, index_scope = token_value(tokens[square_brackets_end_index])
            assert len(index) == 1, f"Error on line {line}"
            index = parse_line(index[0], line, data, scope=index_scope)
            
            collection = parse_line(tokens[:square_brackets_end_index], line, data)
            
            if collection.type in ("CLASS", "FUNCTION"):
                if index.type == "COLLECTION":
                    for v in index.collection:
                        assert v.type in ("DEFINITION", "CONSTANTS", "FUNCTION", "CLASS"), f"Error on line {line}"
                    
                    values = index.data["collection"]
                else:
                    values = [index]
                
                collection = collection.copy()
                
                for i, v in enumerate(values):
                    list(collection.data["scope"]["constants"].values())[i].update(v)
                
                code_value = collection
            else:
                assert index.data_type == data["classes"]["int"], f"Error on line {line}"
                
                code_value = collection[index.data["value"]] if index.name == "NUMBER-LIT" else eat("INDEX", code_lines, collection=collection, index=index)
    elif token_name_value is not None:
        if token_name_value in data["variables"]:
            code_value = data["variables"][token_name_value]
        elif token_name_value in data["constants"]:
            code_value = data["constants"][token_name_value]
        elif token_name_value in data["classes"]:
            code_value = data["classes"][token_name_value]
        elif token_name_value in data["functions"]:
            code_value = data["functions"][token_name_value]
        elif a0:
            code_value = eat("NAME", code_lines, value=token_name_value)
        else:
            raise Exception(f"Error in line {line}")
    elif token_ls_value is not None:
        for p in reversed(parents):
            assert isinstance(p, ParsedCode), f"Error on line {line}"
            assert p.type != "CLASS", f"Error on line {line}"
            
            if "LOOP" in p.type:
                break
        
        code_value = eat("LOOP-STATEMENT", code_lines, s_type=token_ls_value)
    elif token_number_value is not None:
        if token_number_value.startswith("0b"):
            code_value = int(token_number_value, 2)
            is_int = True
        elif token_number_value.startswith("0x"):
            code_value = int(token_number_value, 16)
            is_int = True
        elif token_number_value.startswith("0o"):
            code_value = int(token_number_value, 8)
            is_int = True
        elif (
                "e" in token_number_value and
                token_number_value.count("e") == 1 and
                token_number_value.replace("e", "").isnumeric() and
                (token_number_value.split("e")[0].count(".") <= 1 and token_number_value.split("e")[0].replace(".", "").isnumeric()) and
                token_number_value.split("e")[1].isnumeric()
            ):
            num, exp = token_number_value.split("e")
            
            code_value = float(num[next(i for i, s in enumerate(num) if s != "0"):]) * 10**int(exp[next(i for i, s in enumerate(exp) if s != "0"):])
            is_int = False
        elif token_number_value.count(".") == 1:
            code_value = float(token_number_value)
            is_int = False
        elif token_number_value.count(".") == 0:
            code_value = int(token_number_value)
            is_int = True
        
        if code_value is None:
            raise Exception(f"Error on line {line}")
        
        code_value = eat("NUMBER-LIT", code_lines, value=code_value, data_type=(data["classes"]["int"] if is_int else data["classes"]["float"]))
    elif token_string_value is not None:
        code_value = eat("STRING-LIT", code_lines, value=token_string_value, data_type=data["classes"]["string"])

    SCOPE = prev_scope
    
    if code_value is not None:
        return code_value
    else:
        raise Exception(f"Error on line {line}")

