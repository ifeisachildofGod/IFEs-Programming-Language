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

def _temp_proc_func(parse: list):
    with open(compile_path, "w") as file:
        file.write("\n".join(parse))
    
    print(f'Python transpilation saved to "{compile_path}"')


file_path = sys.argv[1]
compile_path = file_path.removesuffix(".il") + ".py"

with open(file_path) as file:
    print("Compiling")
    
    code = parse_code(tokenize(PREPENSION + file.read() + APPENSION))

print("Linking")

link(code, "py", _temp_proc_func)



