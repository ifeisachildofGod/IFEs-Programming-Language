# import os

from helpers import *
from parsing import *
from tokens import *

test_code = """
get math;

int x = a.b;
float y = 1.1;
x += 1;

string name = "Ife is {x}";
name.replace("f", "d");
name.ife.atu.e()[1] = "11221 dakd";

dict data = ("hello": 2, "he": 2);

forevery (name, value) in data {
    say("${name} is ${value} and \\'you cant do shit about it");
}

if [name == "ife" and x == 2] {
    say("${name} is ife and ${x} is 2");
}

func Info(string name, int age, float accountBalance = 10.0) {
    return say("${name} is ${age} years old and has $${accountBalance} in his bank account");
}

func main(list[string] args) {
    Info("Ife", 16);
}
"""

def _update_data(scope: dict | list | tuple | set, back_track_parent=None):
    if isinstance(scope, dict):
        data = {}
        
        for k, v in scope.items():
            if v is None:
                k_new = _update_data(k, back_track_parent or scope)
                
                if type(k_new) != type(k):
                    data.update(k_new)
                else:
                    data[k_new] = v
            else:
                data[k] = _update_data(v, back_track_parent or scope)
        
        scope = data
    elif isinstance(scope, (list, tuple, set)):
        assert back_track_parent is not None, "Internal Error"
        
        data = scope.__class__([_update_data(v, back_track_parent) for v in scope])
    elif isinstance(scope, str):
        if ";;" in scope:
            assert back_track_parent is not None, "Internal Error"
            
            parent, operation = scope.split(";;")
            
            body = search(parent, back_track_parent)[-1]
            
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

constants = _update_data({
    "@keywords": [
        "func", "class",
        "if", "elseif", "else",
        "forevery", "while",
        "get", "give", "return",
        "in", "at",
        "null"
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
        "@spec_bitwise": {
            "@bitwise_not": "~"
        },
        "@comparators": {
            "@or": "||",
            "@and": "&&",
            
            "@is_equal": "==",
            "@not_equal": "!="
        },
        "@spec_comparators": {
            "@xor": "^^",
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
scope = {
    "@strings": [],
    
    "@curly_brackets": [],
    "@circle_brackets": [],
    "@square_brackets": [],

    "@name": [],
    "@number": [],
    "@reference": [],
    "@collection": [],
    "@pointer": []
}

# linked = []

def tokenize(code: str):
    code = substitute_context(code, "#", "\n", placeholder_func=lambda _: "", strict=False)
    code = substitute_context(code, "/*", "*/", placeholder_func=lambda _: "", strict=False)
    
    code = substitute_strings(code, ["'", '"'], scope["@strings"], lambda i: f"`@strings?{i}`")
    code = substitute_context(code, "{", "}", scope["@curly_brackets"], lambda i: f"{{@curly_brackets?{i}}};", tokenize)
    code = substitute_context(code, "(", ")", scope["@circle_brackets"], lambda i: f"{{@circle_brackets?{i}}}", tokenize)
    code = substitute_context(code, "[", "]", scope["@square_brackets"], lambda i: f"{{@square_brackets?{i}}}", tokenize)
    code = code.replace("`@", "{@")
    code = code.replace("`", "}")
    
    for operator in constants["@operators"].values():
        for symbol in filter(lambda v: len(v.strip()) == 2, operator.values()):
            code = code.replace(symbol, get_token_name(symbol, constants))
    
    for operator in constants["@operators"].values():
        for symbol in filter(lambda v: len(v.strip()) == 1, operator.values()):
            code = code.replace(symbol, get_token_name(symbol, constants))
    
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
                    
                    if v in constants["@substitutions"]:
                        p_holder = get_token_name(constants["@substitutions"][v], constants)
                    elif v in constants["@keywords"]:
                        p_holder = get_token_name(v, constants)
                    else:
                        if is_text:
                            if v not in scope["@name"]:
                                scope["@name"].append(v)
                        elif is_number:
                            if v not in scope["@number"]:
                                scope["@number"].append(v)
                        
                        p_holder = get_token_name(v, scope)
                    
                    replacement_data.append((value_slice, p_holder))
                    
                    is_text = is_number = False
            else:
                assert i != len(code) - 1, "Error"
        elif c == "}":
            continue_reading = True
    
    for value_slice, p_holder in reversed(replacement_data):
        code = slice_text(code, value_slice.start, value_slice.stop, p_holder)
    
    for symbol in constants["@functionality"].values():
        code = code.replace(symbol, get_token_name(symbol, constants))
    
    code_lines = [
        [v.strip().removesuffix("}")
         for v
         in line.strip().split("{")
        ][1:]
        for line
        in code.split(";") if line.strip()
    ]
    
    single_jump_substitution(code_lines, get_token_name(constants["@functionality"]["@dot"], constants, False), "@reference", scope)
    single_jump_substitution(code_lines, get_token_name(constants["@functionality"]["@colon"], constants, False), "@pointer", scope)
    full_cover_substitution(code_lines, get_token_name(constants["@functionality"]["@comma"], constants, False), "@collection", scope)
    
    return code_lines

# def link(tokens: list[list]):
#     return tokens

# @overload
# def kw_get(_: KeyWord, package: Name):
#     with open(scope["@name"][package.id]) as file:
#         for v in reversed(link(tokenize(file.read()))):
#             linked.insert(0, v)

# @overload
# def kw_get(_: KeyWord, package: Reference):
#     with open(os.path.join([find(name, scope) for name in scope["@reference"][package.id]])) as file:
#         for v in reversed(link(tokenize(file.read()))):
#             linked.insert(0, v)

print(*tokenize(test_code), sep="\n")
