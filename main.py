# import os

from helpers import *
from lexer import *
from parser import *

test_code = """
class string() {
    var ife: string = "";
    func replace(var a: string, var b: string) => string {
        return "Hello"
    }
}

func say(var data: string) => null {}

get math;

var x: int = 2;
var y: float = 1.1;

x += 1;

var name: string = "Ife is {x}";
name.replace("f", "d");

var data: list = ["hello", "he"];

forevery (var value: string) in data {
    say("{value} is a value that you cant do shit about it");
}

while (1) {
    break;
}

for (var i: int; i += 1; i < 1) {
    say(i);
}

if (name == "ife" and x == 2) {
    say("{name} is ife and {x} is 2");
}

func Info(var name: string, var age: int, var accountBalance: float = 10.0) => null {
    return say("{name} is {age} years old and has ${accountBalance} in his bank account");
}

func main(const args: list[string]) {
    Info("Ife", 16);
}
"""

code_instructions = []

lines, scope = tokenize(test_code)
print(*lines, sep="\n")
print()
print()

data = {"variables": {"null": ParsedCode(type="NULL", data=None, data_type="NULL")}, "constants": {}, "functions": {}, "classes": {
        "string": ParsedCode(type="CLASS", data={"donor": None}, data_type="CLASS"),
        "int": ParsedCode(type="CLASS", data={"donor": None}, data_type="CLASS"),
        "float": ParsedCode(type="CLASS", data={"donor": None}, data_type="CLASS"),
        "list": ParsedCode(type="CLASS", data={"donor": None}, data_type="CLASS"),
        }}

for line_number, line in enumerate(lines):
    parse_line(line, line_number + 1, data, code_instructions, scope, [])

print(*[c.type for c in code_instructions], sep="\n")


