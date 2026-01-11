

import os
from typing import Any
from copy import deepcopy
from dataclasses import dataclass

from lexer import *

class UnaccountedException(Exception):
    pass

@dataclass
class ParsedCode:
    type: str
    data: dict[str, Any]
    data_type: "ParsedCode" = None
    parse_id: str = None
    
    def copy(self):
        return ParsedCode(type=self.type, data=self.data, data_type=self.data_type)
    
    def update(self, other: "ParsedCode"):
        self.type = other.type
        self.data = other.data
        self.data_type = other.data_type
        self.parse_id = other.parse_id


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

def token_value(token: str, scope: dict | None = None):
    if scope is None:
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

def parser_get_token_name(symbol: str, with_borders=False):
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


def is_defined(tokens: list[str], data: dict[dict[str, ParsedCode]]):
    try:
        parse_line(tokens, 0, data)
    except UnaccountedException:
        return False
    
    return True


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


def eat(name: str, code_lines: list[ParsedCode] | None = None, data_type=None, parse_id=None, **variables):
    code_line = ParsedCode(name, variables, data_type, parse_id)
    
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
    
    comma_present = parser_get_token_name(",") in tokens
    keyword_present = is_token_type(tokens[0], "@keywords")
    equality_present = parser_get_token_name("=") in tokens
    equality_by_index = next((i for i, t_v in enumerate(tokens) if is_token_type(t_v, "@operators?@equality")), None)
    arith_or_bit_or_comp_value = next(((i, t) for i, t in enumerate(tokens) if is_token_type(t, ("@operators?@arithmetic", "@operators?@bitwise", "@operators?@comparators"))), None)
    spec_bit_or_comp_present = next((True for t in tokens if is_token_type(t, ("@operators?@spec_bitwise", "@operators?@spec_comparators"))), False)
    dot_present = parser_get_token_name(".") in tokens
    circle_brackets_end_index = next((len(tokens) - i - 1 for i, t_v in enumerate(reversed(tokens)) if is_token_type(t_v, "@circle_brackets")), None)
    square_brackets_end_index = next((len(tokens) - i - 1 for i, t_v in enumerate(reversed(tokens)) if is_token_type(t_v, "@square_brackets")), None)
    token_name_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@name") else None
    token_ls_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@loopstatments") else None
    token_number_value = token_value(tokens[0]) if len(tokens) == 1 and is_token_type(tokens[0], "@number") else None
    token_string_value = token_value(tokens[0])[1] if len(tokens) == 1 and is_token_type(tokens[0], "@strings") else None
    
    code_value = None
    
    if comma_present:
        code_value = eat("COLLECTION", code_lines, collection=[parse_line(token, line, data) for token in split_list(tokens, parser_get_token_name(","))])
    elif keyword_present:
        match token_value(tokens[0]):
            case "var":
                if parser_get_token_name("=") in tokens:
                    definition, value = split_list(tokens, parser_get_token_name("="))
                    (_, name), _type = split_list(definition, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg=f"Error on line {line}")
                    assert not is_defined(name, data), f"Error on line {line}"
                    
                    name = token_value(name)
                    value = parse_line(value, line, data)
                    
                    code_value = data["var"][name] = eat("DEFINITION", code_lines, cls=value.data["cls"], data_type=parse_line(_type, line, data), parse_id="var", parent=parents)
                    
                    code_value.data["name"] = name
                    
                    eat("ASSIGNMENT", code_lines, object=code_value, value=value)
                else:
                    (_, name), _type = split_list(tokens, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    assert not is_defined(name, data), f"Error on line {line}"
                    
                    name = token_value(name)
                    
                    code_value = data["var"][name] = eat("DEFINITION", type=parse_line(_type, line, data), parse_id="var", parent=parents)
                    
                    code_value.data["name"] = name
            case "const":
                if a0 == "FUNC":
                    (_, name), _type = split_list(tokens, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    assert not is_defined(name, data), f"Error on line {line}"
                    
                    name = token_value(name)
                    
                    code_value = data["const"][name] = eat("C-DEFINITION", code_lines, data_type=parse_line(_type, line, data), parse_id="const", parent=parents)
                    
                    code_value.data["name"] = name
                else:
                    definition, value = split_list(tokens, parser_get_token_name("="))
                    (_, name), _type = split_list(definition, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    assert not is_defined(name, data), f"Error on line {line}"
                    
                    name = token_value(name)
                    
                    code_value = data["const"][name] = eat("C-DEFINITION", code_lines, data_type=parse_line(_type, line, data), parent=parents)
                    
                    code_value.data["name"] = name
                    
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
                    
                    return_type = data["class"]["int"]
                    
                    func_actions = []
                    sub_data = deepcopy(data)
                    action, action_scope = token_value(action)
                    for l, token in enumerate(action):
                        line_val = parse_line(token, line + l + 1, sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)])
                        
                        assert line_val.type != "RETURN", f"Error on line {line}"
                    
                    data["func"][name] = code_value = eat("MAIN", code_lines, params=parse_line(parameters, line, sub_data, scope=param_scope, parents=parents + [(code_value, return_type)], a0="FUNC"), action=func_actions)
                else:
                    if len(split_list(tokens, parser_get_token_name("=>"))[0]) == 3:
                        (_, name, parameters), part2 = split_list(tokens, parser_get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, data), part2[-1]
                        
                        generic_collection = {}
                    elif len(split_list(tokens, parser_get_token_name("=>"))[0]) == 4:
                        (_, name, generics, parameters), part2 = split_list(tokens, parser_get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, data), part2[-1]
                        
                        assert_token_type(generics, "@square_brackets", error_msg = f"Error on line {line}")
                        generics, generics_scope = token_value(generics)
                        assert len(generics) == 1, f"Error on line {line}"
                        generic_collection = {token_value(g_type): token_value(g_token, generics_scope) for g_type, g_token in split_list(generics[0], parser_get_token_name(","))}
                    
                    assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    
                    name = token_value(name)
                    
                    data["func"][name] = code_value = eat("FUNCTION", code_lines, generic_type_ids=list(generic_collection), return_type=return_type, parse_id="func", data_type=data["class"]["callable"])
                    
                    sub_data = deepcopy(data)
                    
                    assert_token_type(parameters, "@circle_brackets", error_msg = f"Error on line {line}")
                    parameters, param_scope = token_value(parameters)
                    assert len(parameters) <= 1, f"Error on line {line}"
                    parameters = parse_line(parameters[0], line, sub_data, scope=param_scope, parents=parents + [(code_value, return_type)], a0="FUNC") if parameters else None
                    
                    for g_t, g in generic_collection.items():
                        sub_data[g_t][g] = eat("GENERIC", [], data_type="GENERIC")
                
                    func_actions = []
                    action, action_scope = token_value(action)
                    for l, token in enumerate(action):
                        line_val = parse_line(token, line + l + 1, sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)])
                        
                        if line_val.type == "RETURN":
                            break
                        elif l == len(action) - 1 and return_type.type == "VOID":
                            parse_line([parser_get_token_name("return"), "@name?0"], line + l + 1, sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)])
                    
                    code_value.data["name"] = name
                    code_value.data["params"] = parameters
                    code_value.data["action"] = func_actions
                    code_value.data["scope"] = sub_data
            case "class":
                if len(tokens) == 4:
                    _, name, donor, action = tokens
                    generic_collection = {}
                elif len(tokens) == 5:
                    _, name, generics, donor, action = tokens
                    
                    assert_token_type(generics, "@square_brackets", error_msg = f"Error on line {line}")
                    generics, generics_scope = token_value(generics)
                    assert len(generics) == 1, f"Error on line {line}"
                    generic_collection = {token_value(g_type): token_value(g_token, generics_scope) for g_type, g_token in split_list(generics[0], parser_get_token_name(","))}
                
                assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                
                assert_token_type(donor, "@circle_brackets", error_msg = f"Error on line {line}")
                donor, donor_scope = token_value(donor)
                assert len(donor) <= 1, f"Error on line {line}"
                donor = parse_line(donor[0], line, data, scope=donor_scope) if donor else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                name = token_value(name)
                
                code_value = data["class"][name] = eat("CLASS", code_lines, generic_type_ids=list(generic_collection), donor=donor, parse_id="class", data_type=data["class"]["callable"])
                
                sub_data = deepcopy(data)
                
                for g_t, g in generic_collection.items():
                    sub_data[g_t][g] = eat("GENERIC", [], data_type="GENERIC")
                
                class_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parsed_line = parse_line(token, line + l + 1, sub_data, class_actions, action_scope, parents=parents + [code_value])
                    
                    assert parsed_line.type in ("CLASS", "FUNCTION", "DEFINITION", "IMPORT"), f"Error on line {line + l + 1}"
                
                code_value.data["name"] = name
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
                (_, variables), part2 = split_list(tokens, parser_get_token_name("in"))
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
                g_data = split_list(tokens[1:], parser_get_token_name("as"))
                if len(g_data) == 1:
                    p_tokens, m_alias = g_data[0], None
                elif len(g_data) == 2:
                    p_tokens, m_alias = g_data
                
                path = os.path.join(*[(token_value(token[0])[1] if isinstance(token_value(token[0]), tuple) else token_value(token[0])) for token in split_list(p_tokens, parser_get_token_name("/"))]) + ".il"
                
                with open(path) as file:
                    import_parse = parse_code(tokenize(file.read()))
                
                assert import_parse[-1].type == "EXPORT", f"Error on line {line}"
                import_val = import_parse[-1].data["cls"]
                assert import_val.type in ("CLASS", "FUNCTION", "DEFINITION", "CONSTANTS"), f"Error on line {line}"
                
                g_name = m_alias or import_val.data["name"]
                
                data[import_val.parse_id][g_name] = import_val
                
                import_val.data["name"] = g_name
                
                code_value = eat("SKIP")
            case "give":
                eat("EXPORT", code_lines, cls=parse_line(tokens[1:], line, data, parents=[]))
                
                code_value = eat("EOF")
            
            case "return":
                return_value = parse_line(tokens[1:], line, data)
                anticipated_value = next(type_hierachy[1] for type_hierachy in reversed(parents) if isinstance(type_hierachy, (list, tuple)))
                
                assert (return_value.data_type == anticipated_value) or (return_value == anticipated_value), f"Error on line {line}"
                
                code_value = eat("RETURN", code_lines, value=return_value)
        
        if code_value is None:
            raise Exception("Internal Error")
    elif equality_present:
        definition, value = split_list(tokens, parser_get_token_name("="))
        
        assert is_defined(definition, data), f"Error on line {line}"
        
        value = parse_line(value, line, data)
        definition = parse_line(definition, line, data)
        
        assert definition.data_type == value.data_type, f"Error on line {line}"
        
        definition.data["cls"] = value.data["cls"]
        
        code_value = eat("ASSIGNMENT", code_lines, object=definition, value=value)
    elif equality_by_index is not None:
        assert len([i for i, t_v in enumerate(tokens) if is_token_type(t_v, "@operators?@equality")]), f"Error on line {line}"
        
        equality_type = tokens[equality_by_index]
        
        variable, value = split_list(tokens, equality_type)
        
        if "@implicit_equals" in equality_type:
            value = parse_line(value, line, data)
            
            assert len(variable) == 1, f"Error on line {line}"
            
            name = variable[0]
            
            assert not is_defined(name, data), f"Error on line {line}"
            assert is_token_type(name, "@name"), f"Error on line {line}"
            
            name = token_value(name)
            
            code_value = data["var"][name] = eat("DEFINITION", code_lines, cls=value.data["cls"], type=value.data_type)
            
            code_value.data["name"] = name
            
            eat("ASSIGNMENT", code_lines, object=data["var"][name], value=value)
        else:
            assert is_defined(variable, data), f"Error on line {line}"
            
            code_value = parse_line(variable + [parser_get_token_name("=")] + variable + [parser_get_token_name(token_value(equality_type).replace("=", ""))] + value, line, data) # parse_line(value, line, data)
            
            # code_value = eat("ASSIGNMENT-BY", code_lines, object=parse_line(variable, line, data), value=value)
    elif arith_or_bit_or_comp_value is not None:
        first_arith_bit_index, arith_bit_token = arith_or_bit_or_comp_value
        
        ls = parse_line(tokens[:first_arith_bit_index], line, data)
        rs = parse_line(tokens[first_arith_bit_index + 1:], line, data)
        
        code_value = eat("BIN-OP", code_lines, left_side=ls, right_side=rs, op_type=arith_bit_token, cls=deepcopy(ls.data["cls"]), data_type=ls.data_type) # Update later
    elif spec_bit_or_comp_present:
        assert is_token_type(tokens[0], ("@operators?@spec_bitwise", "@operators?@spec_comparators")), f"Error on line {line}"
        
        target = parse_line(tokens[1:], line, data)
        
        code_value = eat("SPEC-OP", code_lines, cls=deepcopy(target.data["cls"]), data_type=target.data_type, target=target, op_type=tokens[0])
    elif dot_present:
        ref = None
        line_content = split_list(tokens, parser_get_token_name("."))
        
        for obj in line_content:
            ref = parse_line(obj, line, ref.data["cls"].data["scope"] if ref is not None else data, code_lines, scope, a0=True)
        
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
            
            assert callable_value.data_type == data["class"]["callable"], f"Error on line {line}"
            
            code_value = eat("CALL", code_lines, cls=deepcopy(callable_value), parameters=parameters, data_type=callable_value.data["return_type"])
    elif square_brackets_end_index is not None and (circle_brackets_end_index is None or square_brackets_end_index > circle_brackets_end_index):
        if len(tokens) == 1:
            ls, ls_scope = token_value(tokens[0])
            assert len(ls) <= 1, f"Error on line {line}"
            collection = parse_line(ls[0], line, data, scope=ls_scope) if ls else None
            
            code_value = eat("LIST", code_lines, collection=collection, cls=deepcopy(data["class"]["list"]), data_type=data["class"]["list"])
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
                    list(collection.data["scope"][collection.data["generic_type_ids"][i]].values())[i].update(v)
                
                code_value = collection
            else:
                assert collection.type == "LIST", f"Error on line {line}"
                assert index.data_type == data["class"]["int"], f"Error on line {line}"
                
                code_value = eat("INDEX", code_lines, collection=collection, cls=list(collection.data["scope"]["class"].values())[0], index=index)
    elif token_name_value is not None:
        if token_name_value in data["var"]:
            code_value = data["var"][token_name_value]
        elif token_name_value in data["const"]:
            code_value = data["const"][token_name_value]
        elif token_name_value in data["class"]:
            code_value = data["class"][token_name_value]
        elif token_name_value in data["func"]:
            code_value = data["func"][token_name_value]
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
        
        code_value = eat("NUMBER-LIT", code_lines, value=code_value, cls=deepcopy(data["class"]["int"] if is_int else data["class"]["float"]), data_type=(data["class"]["int"] if is_int else data["class"]["float"]))
    elif token_string_value is not None:
        code_value = eat("STRING-LIT", code_lines, value=token_string_value, cls=deepcopy(data["class"]["string"]), data_type=data["class"]["string"])

    SCOPE = prev_scope
    
    if isinstance(code_value, ParsedCode):
        return code_value
    else:
        raise UnaccountedException(f"Error on line {line}")


def parse_code(token_data: tuple[list[list[str]], dict[str, list[str]]]):
    code_instructions = []
    
    lines, scope = token_data
    
    c = ParsedCode(type="CLASS", data={"donor": None}, data_type="CLASS")
    v = ParsedCode(type="VOID", data=None, data_type="CLASS")
    data = {"var": {"null": ParsedCode(type="NULL", data=None, data_type="NULL"), "void": v}, "const": {}, "func": {}, "class": {
            "string": ParsedCode(type="CLASS", data={"donor": None, "scope": {"var": {"null": ParsedCode(type="NULL", data=None, data_type="NULL"), "void": v}, "const": {}, "func": {
                "replace": ParsedCode(type="FUNCTION", data={"donor": None}, data_type=c)
                }, "class": {
                "string": ParsedCode(type="CLASS", data={"donor": None, "scope": {}}, data_type=c),
                "int": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
                "float": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
                "list": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
                "callable": c,
                }}}, data_type=c),
            "int": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
            "float": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
            "list": ParsedCode(type="CLASS", data={"donor": None}, data_type=c),
            "callable": c,
            }}

    for line_number, line in enumerate(lines):
        line_val = parse_line(line, line_number + 1, data, code_instructions, scope, [])
        
        if line_val.type == "EOF":
            break
    
    return code_instructions


