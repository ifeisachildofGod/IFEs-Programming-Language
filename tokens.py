from dataclasses import dataclass


@dataclass
class _Token:
    id: int

class KeyWord(_Token): pass

class Name(_Token): pass
class Reference(_Token): pass
class Number(_Token): pass
class String(_Token): pass

class Context(_Token): pass
class SquareBrace(_Token): pass
class CircleBrace(_Token): pass
class CurlyBrace(_Token): pass

class FuncEquality(_Token): pass
class FuncComma(_Token): pass
class FuncColon(_Token): pass

class Operator(_Token): pass
class OperatorBy(_Token): pass

