# import os

from helpers import *
from lexer import *
from parser import *

test_code = """
class list[class _T] () {}
class string() {
    func replace(var a: string, var b: string) => void {}
}

get test_module;

func say(var data: string) => null {}

var x: int = 2;
var y: float = 1.1;

x += 1;

var name: string = "Ife is {x}";
name.replace("f", "d");

var data: list[string] = ["hello", "he"];

forevery (var value: string) in data {
    say("{value} is a value that you cant do shit about it");
}

while (1) {
    break;
}

for (var i: int = 0; i += 1; i < 1) {
    say(i);
}

if (name == "ife" and x == 2) {
    say("{name} is ife and {x} is 2");
}

func main(const args: list[string]) {
    Info("Ife", 16);
}
"""


print(*[p.type for p in parse_code(tokenize(test_code))], sep="\n\n")
