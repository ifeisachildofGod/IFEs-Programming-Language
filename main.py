# import os
import sys

from lexer import *
from parser import *
from linker import *


PREPENSION = f"""
using stdlib;
class main(){"{"}
"""

APPENSION = f"""
{"}"}
"""


file_path = sys.argv[1]
compile_path = file_path.removesuffix(".il") + ".py"

with open(file_path) as file:
    code = parse_code(tokenize(PREPENSION + file.read() + APPENSION))

print("Compilation complete")

with open(compile_path, "w") as file:
    file.write(link(code, "py"))

print(f'Code compiled to "{compile_path}"')

