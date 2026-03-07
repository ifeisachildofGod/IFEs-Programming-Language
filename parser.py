
import os
from copy import deepcopy

from lexer import *
from exceptions import *
from parse_token import ParsedCode


PARSED_CODE: list[ParsedCode] = []
SCOPE: dict[str, list] | None = None
DATA: dict[str, dict[str, ParsedCode]] = {}
CONTEXT: list[dict[str, dict[str, ParsedCode]]] = []
CONTEXT_CACHE: dict[str, dict[str, dict[str, ParsedCode]]] = {}


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

def _join_dict(parent: dict, overrider: dict, override_keys: list[str] | None = None):
    new_dict = parent.copy()
    
    if override_keys is None:
        new_dict.update(overrider)
    else:
        for key in override_keys:
            new_dict[key] = new_dict.get(key, {})
            
            new_dict[key].update(overrider.get(key, new_dict.get(key, {})))
    
    return new_dict

def _get_parent_context(parent_context: list[dict[str, dict[str, ParsedCode]]]):
    global CONTEXT_CACHE
    
    p_context_key = "".join([str(v) for v in parent_context])
    
    if p_context_key in CONTEXT_CACHE:
        total_context = CONTEXT_CACHE[p_context_key]
    else:
        total_context = {}
        
        for datas in parent_context:
            total_context = _join_dict(total_context, datas, ["var", "class", "func", "const"])
        
        CONTEXT_CACHE[p_context_key] = total_context
    
    return total_context


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


def is_defined(tokens: list[str], context: dict[dict[str, ParsedCode]]):
    try:
        parse_line(tokens, 0, [context])
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
    code_line = ParsedCode(CONTEXT, name, variables, data_type, parse_id)
    
    if code_lines is not None:
        code_lines.append(code_line)
    
    return code_line


def parse_line(tokens: list[str], line: int, parent_context: list[dict[str, dict[str, ParsedCode]]], context: dict[str, dict[str, ParsedCode]] | None = None, code_lines: list[ParsedCode] | None = None, scope: dict[str, list] | None = None, parents: list | None=None, a0=None):
    global SCOPE, CONTEXT, CONTEXT_CACHE
    
    prev_scope = SCOPE
    
    if scope is not None:
        SCOPE = scope
    if code_lines is None:
        code_lines = []
    
    if not context:
        context = _join_dict({}, {}, ["var", "class", "func", "const"])
    
    total_context = _get_parent_context(parent_context)
    total_context = _join_dict(total_context, context, ["var", "class", "func", "const"])
    
    CONTEXT = parent_context + [context]
    
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
        code_value = eat("COLLECTION", code_lines, collection=[parse_line(token, line, parent_context, a0=a0) for token in split_list(tokens, parser_get_token_name(","))])
    elif keyword_present:
        match token_value(tokens[0]):
            case "var":
                if parser_get_token_name("=") in tokens:
                    definition, value = split_list(tokens, parser_get_token_name("="))
                    (_, name), _type = split_list(definition, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg=f"Error on line {line}")
                    assert not is_defined(name, context), f"Error on line {line}"
                    
                    name = token_value(name)
                    data_type = parse_line(_type, line, parent_context, context, a0="//FUNC//")
                    value = parse_line(value, line, parent_context, context, a0="//FUNC//")
                    
                    v_def = context["var"][name] = eat("DEFINITION", None, data_type=data_type, parse_id="var", parent=parents)
                    code_value = eat("ASSIGNMENT", code_lines, object=v_def, value=value)
                    
                    v_def.variables["name"] = name
                    code_value.variables["name"] = name
                else:
                    (_, name), _type = split_list(tokens, parser_get_token_name(":"))
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    assert not is_defined(name, context), f"Error on line {line}"
                    
                    name = token_value(name)
                    
                    code_value = context["var"][name] = eat("DEFINITION", data_type=parse_line(_type, line, parent_context, context, a0="//FUNC//"), parse_id="var", parent=parents)
                    
                    code_value.variables["name"] = name
            case "const":
                if a0 == ("//FUNC//", "Param") and parser_get_token_name("=") not in tokens:
                    value = None
                    (_, name), _type = split_list(tokens, parser_get_token_name(":"))
                else:
                    definition, v_token = split_list(tokens, parser_get_token_name("="))
                    (_, name), _type = split_list(definition, parser_get_token_name(":"))
                
                    value = parse_line(v_token, line, parent_context, context)
                
                assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                assert not is_defined(name, context), f"Error on line {line}"
                
                name = token_value(name)
                
                c_def = context["const"][name] = eat("C_DEFINITION", None, data_type=parse_line(_type, line, parent_context, context, a0="//FUNC//"), parse_id="const", parent=parents)
                code_value = eat("C_ASSIGNMENT", code_lines, data_type=parse_line(_type, line, parent_context, context, a0="//FUNC//"), object=c_def, value=value)
                
                c_def.variables["name"] = name
                code_value.variables["name"] = name
                
            case "func":
                func_actions = []
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                if len(tokens) == 4:
                    _, name, parameters, action = tokens
                    
                    return_type = total_context["class"]["int"]
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    name = token_value(name)
                    assert name == "main", f"Error on line {line}"
                    
                    assert_token_type(action, "@curly_brackets", error_msg=f"Error on line {line}")
                    action, action_scope = token_value(action)
                    
                    for l, token in enumerate(action):
                        line_val = parse_line(
                            token,
                            line + l + 1,
                            parent_context + [context],
                            sub_data,
                            func_actions,
                            action_scope,
                            parents=parents + [(code_value, return_type)],
                            a0="//FUNC//"
                        )
                        
                        assert line_val.type != "RETURN", f"Error on line {line}, main function does not return anything"
                    
                    generic_type_ids=[]
                else:
                    if len(split_list(tokens, parser_get_token_name("=>"))[0]) == 3:
                        (_, name, parameters), part2 = split_list(tokens, parser_get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, parent_context, context, a0="//FUNC//"), part2[-1]
                        
                        generic_collection = {}
                    elif len(split_list(tokens, parser_get_token_name("=>"))[0]) == 4:
                        (_, name, generics, parameters), part2 = split_list(tokens, parser_get_token_name("=>"))
                        return_type, action = parse_line(part2[:-1], line, parent_context, context, a0="//FUNC//"), part2[-1]
                        
                        assert_token_type(generics, "@square_brackets", error_msg = f"Error on line {line}")
                        generics, generics_scope = token_value(generics)
                        assert len(generics) == 1, f"Error on line {line}"
                        generic_collection = {token_value(g_type): token_value(g_token, generics_scope) for g_type, g_token in split_list(generics[0], parser_get_token_name(","))}
                    
                    assert_token_type(name, "@name", error_msg = f"Error on line {line}")
                    name = token_value(name)
                
                    generic_type_ids = list(generic_collection)
                    
                    for g_t, g in generic_collection.items():
                        sub_data[g_t][g] = eat("GENERIC", [], data_type="GENERIC")
                    
                    assert_token_type(action, "@curly_brackets", error_msg=f"Error on line {line}")
                    action, action_scope = token_value(action)
                    
                    for l, token in enumerate(action):
                        line_val = parse_line(
                            token,
                            line + l + 1,
                            parent_context + [context],
                            sub_data,
                            func_actions,
                            action_scope,
                            parents=parents + [(code_value, return_type)],
                            a0="//FUNC//"
                        )
                        
                        if line_val.type == "RETURN":
                            break
                        elif l == len(action) - 1 and return_type.type == "VOID":
                            parse_line([parser_get_token_name("return"), "@name?0"], line + l + 1, parent_context + [context], sub_data, func_actions, action_scope, parents=parents + [(code_value, return_type)], a0="//FUNC//")
                
                context["func"][name] = code_value = eat(
                    "FUNCTION",
                    code_lines,
                    parse_id="func",
                    scope=sub_data,
                    action=func_actions,
                    data_type=total_context["class"]["callable"],
                    generic_type_ids=generic_type_ids,
                    return_type=return_type,
                )
                
                assert_token_type(parameters, "@circle_brackets", error_msg = f"Error on line {line}")
                parameters, param_scope = token_value(parameters)
                assert len(parameters) <= 1, f"Error on line {line}"
                
                parameters = parse_line(
                        parameters[0],
                        line,
                        parent_context + [context],
                        sub_data,
                        scope=param_scope,
                        parents=parents + [(code_value, return_type)],
                        a0=("//FUNC//", "Param")
                    ) if parameters else None

                code_value.variables["name"] = name
                code_value.variables["params"] = parameters
            case "class":
                assert len(tokens) in (4, 5), f"Error no line {line}, Improper class syntax"
                
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
                donor = parse_line(donor[0], line, parent_context, context, scope=donor_scope) if donor else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                name = token_value(name)
                
                code_value = context["class"][name] = eat(
                    "CLS",
                    code_lines,
                    generic_type_ids=list(generic_collection),
                    donor=donor,
                    parse_id="class",
                    data_type=total_context["class"]["callable"]
                )
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                for g_t, g in generic_collection.items():
                    sub_data[g_t][g] = eat("GENERIC", [], data_type="GENERIC")
                
                class_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parsed_line = parse_line(token, line + l + 1, parent_context + [context], sub_data, class_actions, action_scope, parents=parents + [code_value])
                    
                    assert parsed_line.type in ("CLS", "FUNCTION", "DEFINITION", "IMPORT", "IGNORE"), f"Error on line {line + l + 1}"
                
                code_value.variables["name"] = name
                code_value.variables["action"] = class_actions
                code_value.variables["scope"] = sub_data
                code_value.variables["return_type"] = code_value
            
            case "for":
                assert a0 == "//FUNC//", f"Error on line {line}: for loops are not allowed outside functions"
                
                _, conditions, action = tokens
                
                assert_token_type(conditions, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                loop_conditions = []
                conditions, condition_scope = token_value(conditions)
                for l, token in enumerate(conditions):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, loop_conditions, condition_scope, a0=a0)
                
                code_value = eat("FOR-LOOP", code_lines, conditions=loop_conditions)
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, loop_actions, action_scope, parents=parents + [code_value], a0=a0)
                
                code_value.variables["action"] = loop_actions
            case "forevery":
                assert a0 == "//FUNC//", f"Error on line {line}: forevery loops are not allowed outside functions"
                
                (_, variables), part2 = split_list(tokens, parser_get_token_name("in"))
                collection, action = part2[:-1], part2[-1]
                
                assert_token_type(variables, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                variables, variables_scope = token_value(variables)
                assert len(variables) == 1, "Error"
                variables = parse_line(variables[0], line, parent_context + [context], sub_data, scope=variables_scope)
                
                code_value = eat("FOR-EVERY-LOOP", code_lines, variables=variables, collection=parse_line(collection, line, parent_context, context))
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, loop_actions, action_scope, parents=parents + [code_value], a0=a0)
                
                code_value.variables["action"] = loop_actions
            case "while":
                assert a0 == "//FUNC//", f"Error on line {line}: while loops are not allowed outside functions"
                
                _, condition, action = tokens
                
                assert_token_type(condition, "@circle_brackets", error_msg = f"Error on line {line}")
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                condition, condition_scope = token_value(condition)
                assert len(condition) == 1, "Error"
                condition = condition[0]
                
                code_value = eat("WHILE-LOOP", code_lines, condition=parse_line(condition, line, parent_context + [context], sub_data, scope=condition_scope, a0=a0))
                
                loop_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, loop_actions, action_scope, parents=parents + [code_value], a0=a0)
                
                code_value.variables["action"] = loop_actions
            
            case "if":
                assert a0 == "//FUNC//", f"Error on line {line}: if statements are not allowed outside functions"
                
                _, condition, action = tokens
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                assert_token_type(condition, "@circle_brackets", error_msg = f"Error on line {line}")
                condition, condition_scope = token_value(condition)
                assert len(condition) <= 1, f"Error on line {line}"
                condition = parse_line(condition[0], line, parent_context + [context], sub_data, scope=condition_scope, a0=a0) if condition else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("IF", code_lines, condition=condition)
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, conditonal_actions, action_scope, parents=parents + [code_value], a0=a0)
                
                code_value.variables["action"] = conditonal_actions
            case "elseif":
                assert code_lines[-1].type in ("IF", "ELSE-IF")
                
                _, condition, action = tokens
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                assert_token_type(condition, "cicle_brackets", error_msg = f"Error on line {line}")
                condition, condition_scope = token_value(condition)
                assert len(condition) == 1, f"Error on line {line}"
                condition = parse_line(condition[0], line, parent_context + [context], sub_data, scope=condition_scope, a0="//FUNC//") if condition else None
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("ELSE-IF", code_lines, parent=code_lines[-1], condition=condition)
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], sub_data, conditonal_actions, action_scope, parents=parents + [code_value], a0="//FUNC//")
                
                code_value.variables["action"] = conditonal_actions
            case "else":
                assert code_lines[-1].type in ("IF", "ELSE-IF")
                
                _, action = tokens
                
                assert_token_type(action, "@curly_brackets", error_msg = f"Error on line {line}")
                
                code_value = eat("ELSE", code_lines, parent=code_lines[-1])
                
                sub_data = _join_dict({}, {}, ["var", "class", "func", "const"])
                
                conditonal_actions = []
                action, action_scope = token_value(action)
                for l, token in enumerate(action):
                    parse_line(token, line + l + 1, parent_context + [context], conditonal_actions, sub_data, action_scope, parents=parents + [code_value], a0="//FUNC//")
                
                code_value.variables["action"] = conditonal_actions
            
            case "module":
                g_data = split_list(tokens[1:], parser_get_token_name("as"))
                if len(g_data) == 1:
                    p_tokens, m_alias = g_data[0], None
                elif len(g_data) == 2:
                    p_tokens, m_alias = g_data
                
                path = os.path.join(*[(token_value(token[0])[1] if isinstance(token_value(token[0]), tuple) else token_value(token[0])) for token in split_list(p_tokens, parser_get_token_name("/"))]) + ".il"
                
                with open(path) as file:
                    import_parse = parse_code(tokenize(file.read()))
                
                assert import_parse[-1].type == "EXPORT", f"Error on line {line}"
                import_val = import_parse[-1].variables["value"]
                assert import_val.type in ("CLS", "FUNCTION", "DEFINITION", "CONSTANTS"), f"Error on line {line}"
                
                g_name = m_alias or import_val.variables["name"]
                
                context[import_val.parse_id][g_name] = import_val
                
                import_val.variables["name"] = g_name
                
                code_value = eat("IGNORE")
            case "export":
                eat("EXPORT", code_lines, value=parse_line(tokens[1:], line, parent_context, context, parents=[], a0="//FUNC//"))
                
                code_value = eat("EOF")
            case "using":
                path = os.path.join(*[(token_value(token[0])[1] if isinstance(token_value(token[0]), tuple) else token_value(token[0])) for token in split_list(tokens[1:], parser_get_token_name("/"))]) + ".il"
                name = path.split("/")[-1].replace(".il", "")
                
                with open(path) as file:
                    using_parse = parse_code(tokenize(f"class {name}() {'{'}" + file.read() + "}"))
                    
                    action = using_parse[0].variables["action"]
                    scope = using_parse[0].variables["scope"]
                    
                    code_lines.extend(action)
                    
                    new_context = _join_dict(context, scope, ["var", "class", "func", "const"])
                    
                    context.clear()
                    context.update(new_context)
                
                code_value = eat("IGNORE")
            
            case "return":
                assert a0 == "//FUNC//", f"Error on line {line}: return statements are not allowed outside functions"
                
                return_value = parse_line(tokens[1:], line, parent_context, context)
                anticipated_value = next(type_hierachy[1] for type_hierachy in reversed(parents) if isinstance(type_hierachy, (list, tuple)))
                
                assert (return_value.data_type == anticipated_value) or (return_value == anticipated_value), f"Error on line {line}"
                
                code_value = eat("RETURN", code_lines, value=return_value)
        
        if code_value is None:
            raise Exception("Internal Error")
    elif equality_present:
        assert a0 == "//FUNC//", f"Error on line {line}: statements are not allowed outside functions"
        
        definition, value = split_list(tokens, parser_get_token_name("="))
        
        assert is_defined(definition, context), f"Error on line {line}"
        
        value = parse_line(value, line, parent_context, context, a0="//FUNC//")
        definition = parse_line(definition, line, parent_context, context, a0="//FUNC//")
        
        assert definition.data_type == value.data_type, f"Error on line {line}"
        
        code_value = eat("ASSIGNMENT", code_lines, object=definition, value=value)
    elif equality_by_index is not None:
        assert a0 == "//FUNC//", f"Error on line {line}: statements are not allowed outside functions"
        
        assert len([i for i, t_v in enumerate(tokens) if is_token_type(t_v, "@operators?@equality")]), f"Error on line {line}"
        
        equality_type = tokens[equality_by_index]
        
        variable, value = split_list(tokens, equality_type)
        
        if "@implicit_equals" in equality_type:
            value = parse_line(value, line, parent_context, context, a0="//FUNC//")
            
            assert len(variable) == 1, f"Error on line {line}"
            
            name = variable[0]
            
            assert not is_defined(name, context), f"Error on line {line}"
            assert is_token_type(name, "@name"), f"Error on line {line}"
            
            name = token_value(name)
            
            code_value = context["var"][name] = eat("DEFINITION", code_lines, type=value.data_type)
            
            code_value.variables["name"] = name
            
            eat("ASSIGNMENT", code_lines, object=total_context["var"][name], value=value)
        else:
            assert is_defined(variable, context), f"Error on line {line}"
            
            code_value = parse_line(variable + [parser_get_token_name("=")] + variable + [parser_get_token_name(token_value(equality_type).replace("=", ""))] + value, line, parent_context, context, a0="//FUNC//") # parse_line(value, line, parent_context, context)
            
            # code_value = eat("ASSIGNMENT-BY", code_lines, object=parse_line(variable, line, parent_context, context), value=value)
    elif arith_or_bit_or_comp_value is not None:
        assert a0 == "//FUNC//", f"Error on line {line}: arithmetic operations are not allowed outside functions"
        
        first_arith_bit_index, arith_bit_token = arith_or_bit_or_comp_value
        
        ls = parse_line(tokens[:first_arith_bit_index], line, parent_context, context, a0="//FUNC//")
        rs = parse_line(tokens[first_arith_bit_index + 1:], line, parent_context, context, a0="//FUNC//")
        
        code_value = eat("BIN-OP", code_lines, left_side=ls, right_side=rs, op_type=arith_bit_token, data_type=ls.data_type) # Update later
    elif spec_bit_or_comp_present:
        assert a0 == "//FUNC//", f"Error on line {line}: bitwise operations are not allowed outside functions"
        
        assert is_token_type(tokens[0], ("@operators?@spec_bitwise", "@operators?@spec_comparators")), f"Error on line {line}"
        
        target = parse_line(tokens[1:], line, parent_context, context, a0="//FUNC//")
        
        code_value = eat("SPEC-OP", code_lines, data_type=target.data_type, target=target, op_type=tokens[0])
    elif dot_present:
        assert a0 == "//FUNC//", f"Error on line {line}: statements are not allowed outside functions"
        
        references = []
        
        ref = None
        line_content = split_list(tokens, parser_get_token_name("."))
        
        for obj in line_content:
            ref = parse_line(obj, line, parent_context, ref.data_type.variables["scope"] if ref is not None else context, code_lines, scope, a0="//FUNC//")
            references.append(ref)
        
        code_value = eat("REF", code_lines, data_type=ref.data_type, reference_line=references)
    elif circle_brackets_end_index is not None and (square_brackets_end_index is None or circle_brackets_end_index > square_brackets_end_index):
        assert a0 == "//FUNC//", f"Error on line {line}: statements are not allowed outside functions"
        
        if len(tokens) == 1:
            value, value_scope = token_value(tokens[0])
            assert len(value) == 1, f"Error on line {line}"
            
            code_value = parse_line(value[0], line, parent_context, context, scope=value_scope, a0=a0)
        else:
            assert tokens[:circle_brackets_end_index], f"Error on line {line}"
            
            params, params_scope = token_value(tokens[circle_brackets_end_index])
            assert len(params) <= 1, f"Error on line {line}"
            parameters = parse_line(params[0], line, parent_context, context, scope=params_scope) if params else None
            
            callable_value = parse_line(tokens[:circle_brackets_end_index], line, parent_context, context)
            
            assert "return_type" in callable_value.variables, f"Error on line {line}"
            
            code_value = eat("CALL", code_lines, parameters=parameters, c_class=callable_value, data_type=callable_value.variables["return_type"], a0 = a0)
    elif square_brackets_end_index is not None and (circle_brackets_end_index is None or square_brackets_end_index > circle_brackets_end_index):
        assert a0 == "//FUNC//", f"Error on line {line}: statements are not allowed outside functions"
        
        if len(tokens) == 1:
            ls, ls_scope = token_value(tokens[0])
            assert len(ls) <= 1, f"Error on line {line}"
            collection = parse_line(ls[0], line, parent_context, context, scope=ls_scope, a0="//FUNC//",) if ls else None
            
            if collection is None:
                collection = []
            elif collection.type == "COLLECTION":
                collection = collection.variables["collection"]
            else:
                collection = [collection]
            
            code_value = eat("LIST", code_lines, collection=collection, data_type=total_context["class"]["list"])
        else:
            assert tokens[:square_brackets_end_index], f"Error on line {line}"
            
            index, index_scope = token_value(tokens[square_brackets_end_index])
            assert len(index) == 1, f"Error on line {line}"
            index = parse_line(index[0], line, parent_context, context, scope=index_scope, a0="//FUNC//")
            
            collection = parse_line(tokens[:square_brackets_end_index], line, parent_context, context, a0="//FUNC//")
            
            if collection.type in ("CLS", "FUNCTION"):
                if index.type == "COLLECTION":
                    for v in index.collection:
                        assert v.type in ("DEFINITION", "CONSTANTS", "FUNCTION", "CLS"), f"Error on line {line}"
                    
                    values = index.variables["collection"]
                else:
                    values = [index]
                
                collection = collection.copy()
                
                for i, v in enumerate(values):
                    list(collection.variables["scope"][collection.variables["generic_type_ids"][i]].values())[i].update(v)
                
                code_value = collection
            else:
                assert collection.type == "LIST", f"Error on line {line}"
                assert index.data_type == total_context["class"]["int"], f"Error on line {line}"
                
                code_value = eat("INDEX", code_lines, collection=collection, data_type=list(collection.variables["scope"]["class"].values())[0], index=index)
    elif token_name_value is not None:
        if token_name_value in total_context["var"]:
            code_value = total_context["var"][token_name_value]
        elif token_name_value in total_context["const"]:
            code_value = total_context["const"][token_name_value]
        elif token_name_value in total_context["class"]:
            code_value = total_context["class"][token_name_value]
        elif token_name_value in total_context["func"]:
            code_value = total_context["func"][token_name_value]
        elif a0:
            code_value = eat("NAME", code_lines, value=token_name_value)
        else:
            raise AssertionError(f'Error in line {line}, No such variable as "{token_name_value}"')
    elif token_ls_value is not None:
        for p in reversed(parents):
            assert isinstance(p, ParsedCode), f"Error on line {line}"
            assert p.type != "CLS", f"Error on line {line}"
            
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
        
        code_value = eat("NUMBER_LIT", code_lines, value=code_value, data_type=(total_context["class"]["int"] if is_int else total_context["class"]["float"]))
    elif token_string_value is not None:
        code_value = eat("STRING_LIT", code_lines, value=token_string_value, data_type=total_context["class"]["string"])
    
    SCOPE = prev_scope
    
    if isinstance(code_value, ParsedCode):
        return code_value
    else:
        raise UnaccountedException(f"Error on line {line}")


def parse_code(token_data: tuple[list[list[str]], dict[str, list[str]]]):
    code_instructions: list[ParsedCode] = []
    
    lines, scope = token_data
    
    context = {"class": {}, "var": {}, "func": {}, "const": {}}
    parent_context = {}
    
    c = ParsedCode(scope_data=[parent_context], type="CLS", variables={"name": "callable", "donor": None}, data_type="CLS")
    v = ParsedCode(scope_data=[parent_context], type="VOID", variables=None, data_type="CLS")
    parent_context.update(
        {
            "var": {"null": ParsedCode(scope_data=[parent_context], type="NULL", variables=None, data_type="NULL"), "void": v},
            "const": {},
            "func": {},
            "class": {
                "int": ParsedCode(scope_data=[parent_context], type="CLS", variables={"name": "int", "donor": None}, data_type=c),
                "float": ParsedCode(scope_data=[parent_context], type="CLS", variables={"name": "float", "donor": None}, data_type=c),
                "callable": c,
                }
        }
    )
    
    for line_number, line in enumerate(lines):
        line_val = parse_line(line, line_number + 1, [parent_context], context, code_instructions, scope, [])
        
        if line_val.type == "EOF":
            break
    
    return code_instructions


