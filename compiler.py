from typing import Optional
import torch
from torch import nn
import torch.fx
import importlib

from utils import ExcelLambda


def build_excel_array(tensor: torch.Tensor, digits: Optional[int] = None) -> str:
    if tensor.dim() == 1:
        # TODO: May need to unsqueeze 1 instead
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 2
    return '{' + ';'.join(','.join(str(x.item() if digits is None else round(x.item(), digits)) for x in row) for row in tensor) + '}'

def build_excel_function(node: torch.fx.Node) -> str:
    # print(node.target.name())
    if node.kwargs != {}:
        raise NotImplementedError('kwargs not implemented')
    node_name = node.target.name()
    try:
        mod_name, fn_name = node_name.split('::')
        fn_name = fn_name.replace('.', '_')
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        return fn(node)
    except ModuleNotFoundError:
        raise NotImplementedError(f'Node {node_name} not implemented')

    # Height of an array: =MAX(ROW(I9#))-MIN(ROW(I9#))+1


def compile(model: nn.Module, *args, digits=None, **kwargs) -> str:
    export = torch.export.export(model, args, kwargs)
    export = export.run_decompositions()

    nodes = list(export.graph.nodes)
    inputs_to_parameters = export.graph_signature.inputs_to_parameters
    parameters = export.graph_signature.parameters
    user_inputs = export.graph_signature.user_inputs
    constants = export.tensor_constants
    inputs_to_constants = export.graph_signature.inputs_to_lifted_tensor_constants

    code = ''

    # TODO: If a node is only used once, we can inline it

    # Add the function calls
    for node in reversed(nodes):
        match node.op:
            case 'placeholder':
                pass
            case 'call_function':
                fn_code = build_excel_function(node)
                if isinstance(fn_code, str):
                    fn_code = {node.name: fn_code}
                code = f'LET({",".join(x for item in fn_code.items() for x in item)},{code})'
            case 'output':
                output = node.args[0]
                if len(output[0].meta['val'].shape) > 2:
                    raise NotImplementedError('Multi-dimensional outputs not possible in Excel')
                if len(output) > 1:
                    raise NotImplementedError('Multiple outputs not implemented')
                code = output[0].name


    # Add requested LAMBDAs
    function_nodes = [node for node in nodes if node.op == 'call_function']
    node_names = {node.target.name() for node in function_nodes}
    lambdas: dict[str, ExcelLambda] = {}
    for node_name in node_names:
        try:
            mod_name, fn_name = node_name.split('::')
            fn_name = fn_name.replace('.', '_')
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, f'{fn_name}')
            if hasattr(fn, '_lambdas'):
                for k, v in fn._lambdas.items():
                    lambdas[k] = v.mangled(f'{mod_name}_{fn_name}')
        except ModuleNotFoundError:
            raise NotImplementedError(f'Node {node_name} not implemented')

    if len(lambdas) > 0:
        lambda_names = ','.join(lambdas.keys())
        lambda_args = ','.join(f'LAMBDA({",".join(lm.args)},{lm.code})' for lm in lambdas.values())
        code = f'LAMBDA({lambda_names},{code})({lambda_args})'

    # Add fixed weights and constants as LETs
    try:
        lets = []

        parameters_to_inputs = {v: k for k, v in inputs_to_parameters.items()}
        for parameter in parameters:
            lets.append(parameters_to_inputs[parameter])
            lets.append(build_excel_array(model.state_dict()[parameter], digits=digits))

        constants_to_inputs = {v: k for k, v in inputs_to_constants.items()}
        for constant_name, constant_value in constants.items():
            lets.append(constants_to_inputs[constant_name])
            lets.append(build_excel_array(constant_value, digits=digits))

        if len(lets) > 0:
            code = f'LET({",".join(lets)},{code})'
    except AttributeError:
        print('No parameters added')

    # Add the LAMBDA wrapping
    lambda_args = []
    for user_input in user_inputs:
        lambda_args.append(user_input)

    code = f'LAMBDA({",".join(lambda_args)},{code})'

    code = '=' + code
    return code
