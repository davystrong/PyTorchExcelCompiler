from __future__ import annotations
from typing import Callable, ParamSpec, TypeVar, cast
import re

NAME_CHARS = 'A-Za-z_0-9'
# This is just a simplified regex
VAR_REGEX = re.compile(rf'(?<=\$)[{NAME_CHARS}]+')

class ExcelLambda:
    def __init__(self, code: str, arg: str, *args: str) -> None:
        self.code = code
        self.args = [arg, *args]

    def mangled(self, namespace: str) -> ExcelLambda:
        code = self.code
        args = []
        for arg in self.args:
            new_arg = f'__{namespace}_{arg}'
            # This needs to handle the case where the argument is at the start or the end of the code.
            # A positive lookbehind and lookahead would be perfect here, but the lookbehind can't match variable length strings.
            # Instead, I use a group before the arg and then add that group to the replacement code.
            # I can't use the same technique for the end because of the case where there is only one character between two repeats of an arg.
            # However, I can use a lookahead to match the end of the string because it doesn't require fixed length. 
            code = re.sub(rf'([^A-Za-z_0-9]|^){arg}(?=[^A-Za-z_0-9]|$)', rf'\g<1>{new_arg}', code)
            args.append(new_arg)
        return ExcelLambda(code, *args)

Param = ParamSpec("Param")
RetType = TypeVar("RetType")
def register_lambdas(**kwargs: ExcelLambda | str):
    def inner(fn: Callable[Param, RetType]) -> Callable[Param, RetType]:
        lambdas: dict[str, ExcelLambda] = {}
        for key, value in kwargs.items():
            if hasattr(kwargs, 'code') and hasattr(kwargs, 'args'):
                lambdas[key] = cast(ExcelLambda, value)
            else:
                value = cast(str, value)
                # dict is ordered, set isn't
                lambdas[key] = ExcelLambda(value.replace('$', ''), *dict.fromkeys(VAR_REGEX.findall(value)))
        
        fn._lambdas = lambdas
        return fn
    return inner